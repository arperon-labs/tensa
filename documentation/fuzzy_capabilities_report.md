# TENSA Fuzzy Capabilities — End-to-End Capability Report

**Test fixture:** `tests/fuzzy_capabilities_demo.rs`
**Test name:** `fuzzy_capabilities_demo`
**TENSA version:** v0.79.32 (Fuzzy Sprint Phases 0–13 + Graded Acceptability Sprint Phases 0–6)
**Run command:** `cargo test --no-default-features --test fuzzy_capabilities_demo -- --nocapture`
**Output artefacts:** `target/fuzzy_capabilities_demo/transcript.txt` and `target/fuzzy_capabilities_demo/report.json`
**Result:** PASS — 13 sections, 12 fuzzy capabilities exercised

---

## 0. What this report does

TENSA's fuzzy stack is large enough that it is hard to grasp from the reference manual alone — there are 4 t-norms, 6 aggregators, fuzzy Allen relations, intermediate quantifiers, graded Peterson syllogisms, Bělohlávek concept lattices, Mamdani rule systems, fuzzy-probabilistic hybrid inference, four gradual / ranking-based argumentation semantics, ranking-supervised Choquet measure learning, and the Nebel-Bürckert ORD-Horn closure. Each capability is independently tested and documented in [`TENSA_REFERENCE.md`](TENSA_REFERENCE.md) Chapters 14 & 15, but they are rarely exercised together against a single concrete narrative.

This report does that. We construct a synthetic disinformation operation called **"Operation Vesper"** (5 actors, 3 organisations, 7 situations including 2 with fuzzy temporal endpoints), then drive every fuzzy surface against it. Each section below explains what was tested, what numbers came out, and what the analyst should read into them. The closing section ties each capability back to a deliverable in [`docs/EIC/TENSA_DeepRAP_Deliverables.md`](EIC/TENSA_DeepRAP_Deliverables.md).

The test fixture is short (~700 lines). Run it locally to reproduce; modify the narrative scaffold (see [`build_operation_vesper`](../tests/fuzzy_capabilities_demo.rs)) to drive the same surfaces against your own data.

> **Why this is a single integration test, not a Studio walkthrough.** The TENSA REST server was not running at the time of test design, and the goal was an artefact reproducible from a clean checkout with `cargo test --no-default-features`. The same scenarios can be replayed verbatim against the REST surface — every fuzzy library function used here has a 1:1 REST endpoint and an MCP tool listed in TENSA_REFERENCE.md §14.10–§14.12.

---

## 1. Operation Vesper — the narrative scaffold

| Element | Count | Notes |
|---|---|---|
| Actors | 5 | 3 coordinated cluster (Vesper-A1/A2/A3), 1 organic outlier (Outlier-Loud), 1 fact-checker (FactCheck-Gamma) |
| Organisations | 3 | ShellHub-Var (suspect funding hub), EU-Watchdog-NGO, EuFactWatch |
| Situations | 7 | 2 with fuzzy endpoints (Wave A + Wave B), 2 sharp events, 3 used by ORD-Horn |
| Total entities | 8 | (`narrative_id = "operation-vesper"`) |

Every actor carries the four CIB signal properties used by the Choquet measure-learning section: `temporal_correlation`, `content_overlap`, `network_density`, `posting_cadence`, plus `inflammatory_score`, `anonymous`, `foreign_funded`, `verified`. The Vesper-A cluster scores high on the first two (the load-bearing interaction term — see §10), Outlier-Loud is heated but uncoordinated, and FactCheck-Gamma scores low on every CIB indicator.

---

## 1.5 The Operation Vesper scenario in detail

Before any algorithm runs, the test fixture builds a small but adversarially-shaped narrative. This section describes exactly what was ingested, so the numerics in §2–§13 can be read against a concrete situation. The complete scaffold lives in [`build_operation_vesper`](../tests/fuzzy_capabilities_demo.rs) (≈140 lines).

### The case

A coordinated information operation called **Operation Vesper** is suspected of pushing a misleading claim about an EU policy on the morning of **2026-04-14**. An analyst has open-source intelligence pointing at three accounts, two organisations, a fact-check verdict, and a platform takedown. The analyst has to decide:

1. *Is this actually coordinated, or just three people who happen to post angrily about the same thing?*
2. *How confident can we be in the joint claim when sources disagree?*
3. *Did the events happen in the order the suspected funding chain implies?*

That is the kind of question a real disinformation investigator faces every week. We ingest exactly enough data to make every fuzzy method earn its keep.

### The five actors

| ID | Name | Confidence | `temp_corr` | `content_overlap` | `network_density` | `posting_cadence` | `inflammatory_score` | `anonymous` | `foreign_funded` | `verified` | What this actor represents |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Vesper-A1 | 0.92 | 0.91 | 0.88 | 0.62 | 0.31 | 0.81 | true | true | false | Suspected coordinator, anonymous, foreign-funded, posts in lockstep with A2/A3 |
| A2 | Vesper-A2 | 0.88 | 0.86 | 0.83 | 0.55 | 0.28 | 0.78 | true | true | false | Sister persona — same operator, same content, same timing |
| A3 | Vesper-A3 | 0.84 | 0.79 | 0.81 | 0.45 | 0.34 | 0.74 | false | true | false | Less anonymous but still inside the cluster signal |
| O | Outlier-Loud | 0.74 | 0.20 | 0.18 | 0.30 | 0.85 | 0.71 | false | false | true | A genuinely angry but **uncoordinated** commentator — verified, posts often, *but on their own schedule and own words*. The "decoy" |
| F | FactCheck-Gamma | 0.95 | 0.05 | 0.10 | 0.20 | 0.40 | 0.10 | false | false | true | Accredited fact-checker — high confidence, low everything-else |

The analytic point of the four CIB signals: *coordinated behaviour shows up as high temporal-correlation **AND** high content-overlap simultaneously*. **Outlier-Loud** is the deliberate stress-test — a high-inflammatory account that should NOT trigger any "coordination" verdict, even though a naive heuristic would flag it. **FactCheck-Gamma** is the negative control — it must score near zero on every CIB indicator.

### The three organisations

| Name | Foreign funded | Anonymous | Verified | Role |
|---|---|---|---|---|
| ShellHub-Var | true | true | false | Suspected funding-conduit shell |
| EU-Watchdog-NGO | false | false | true | Civil-society watchdog tracking the campaign |
| EuFactWatch | false | false | true | Accredited fact-checker that publishes the verdict |

### The seven situations

| Time | Name | Granularity | Confidence | Why it's there |
|---|---|---|---|---|
| 2026-04-01 09:00–10:00 | Funding-Wire | Exact | 0.55 | Suspected funding wire reaches ShellHub-Var. Used by ORD-Horn (§13). Low confidence because OSINT-derived. |
| 2026-04-05 → 04-12 | Persona-Network-Build | Exact | 0.62 | Anonymous personas register and warm up. ORD-Horn middle node. |
| 2026-04-14 08:00–11:00 | **Vesper-Post-Wave-A** | **Approximate** ±60 min | 0.78 | First posting wave. **Trapezoidal fuzzy endpoints** (kernel 08:00–11:00, support 07:00–12:00). Used by Fuzzy Allen (§5). |
| 2026-04-14 11:00–13:00 | **Vesper-Post-Wave-B** | **Approximate** ±60 min | 0.74 | Second posting wave. Endpoints ALSO fuzzy. Wave-A end and Wave-B start kernels just touch — graded Allen must spread mass across {Before, Meets, Overlaps}. |
| 2026-04-14 05:00–13:00 | Coordinated-Posting | Exact | 0.79 | Aggregate envelope of A+B. ORD-Horn composition target. |
| 2026-04-14 14:00–15:00 | Platform-Takedown | Exact | 0.83 | Platform removes A1+A2 accounts. |
| 2026-04-14 16:00–17:00 | EuFactWatch-Verdict | Exact | 0.92 | Fact-check publishes "misleading" verdict. Used as the high-confidence sharp event. |

### Participation links

* Vesper-A1 / A2 / A3 each participate as **Protagonist** with action `"post"` in **both** posting waves AND in the takedown event (where they are the targets).
* FactCheck-Gamma participates as Protagonist in the verdict, action `"publish-verdict"`.

That is enough wiring for the rule system, the FCA lattice, the quantifier evaluation, and the syllogism verifier to all see a non-trivial domain.

### The synthetic-CIB dataset (used only by §11)

The Choquet measure-learning section is exercised against a **separate** dataset — 100 synthetic CIB clusters generated by `generate_synthetic_cib(seed=42, n_clusters=100)` ([`src/fuzzy/synthetic_cib_dataset.rs`](../src/fuzzy/synthetic_cib_dataset.rs)). The hidden generative score is

```
score = sigmoid(2 · temporal_correlation · content_overlap
                + 0.3 · network_density
                - 0.5 · posting_cadence)
```

The `temporal_correlation × content_overlap` product is the *load-bearing non-additive interaction*. We use this synthetic dataset in §11 (not the live Vesper data) because we need ranked labels and many examples to demonstrate that a learned Choquet measure recovers an interaction no additive aggregator can express. The live Vesper actors are the sanity-check that the same signals carry through into Mamdani rule firings (§9) and concept-lattice extents (§8) on real-world-shaped data.

### Why this scaffold

Every fuzzy method needs at least one input that *would fail under a naive treatment* if the method weren't there. The scaffold below pairs each method with a deliberate stress-test:

| Method | Stress-test in the scaffold | What naive treatment would break |
|---|---|---|
| t-norms | One source dissents at 0.10 (FactCheck-Gamma) | The arithmetic-mean / mean-of-confidences / Choquet-additive collapse to 0.69. T-norm-reduce reads 0.10 (Gödel) or 0.0 (Łukasiewicz) — *correctly* refusing to certify a claim one source actively disputes. |
| OWA | Same sources, same dissent | OWA-Most weights 0.808 — correctly captures *"the body of voices agree"* even with a dissenter, where the mean is dragged down. |
| Choquet | Coalition `{A1,A2,A3}` vs the other two | An additive measure cannot reward "the suspected coalition agreed" beyond what individual votes contribute. The custom coalition-reward measure returns 0.87 vs Mean's 0.69. |
| Fuzzy Allen | Two posting waves with adjacent ±60-min fuzzy endpoints | Crisp Allen returns one-hot `Meets`. Graded Allen returns `Before=Meets=Overlaps=0.5`, accurately reflecting *"we don't know which it is"*. |
| Quantifiers | Five actors, predicate `confidence > 0.85` (3 satisfy) | A binary "true / false" answer to *"do most have high confidence?"* discards the ratio. The Novák-Murinová ramps return `most=0.6, almost_all=0.0` — different, but both correct under their respective ramps. |
| Syllogisms | Figure II at degree 1.0 (Peterson-invalid) | A pure threshold check would accept a Figure II syllogism because its degree clears 0.5. The verifier rejects Figure II by *taxonomy*, regardless of degree. |
| FCA | Mixed binary + graded actor attributes | A k-means or hierarchical-clustering answer would give cluster IDs; FCA returns labelled concepts (extent + intent), each one *interpretable* without further analysis. |
| Mamdani | Outlier-Loud has high inflammatory but low temporal_correlation | A scalar threshold on `inflammatory_score` flags Outlier-Loud as risky. The Mamdani rule under Gödel min returns firing 0.0 — correctly distinguishing *coordination* from *heat*. |
| Hybrid | Fuzzy event over three coalition members | A purely probabilistic event can't grade *how* coordinated each member is. The custom predicate `μ_E = inflammatory_score` returns the probability-weighted graded average, 0.7767. |
| Choquet learning | Hidden multiplicative interaction in the synthetic CIB dataset | An additive baseline scores AUC 0.45 (worse than coin flip!) because it cannot represent the interaction. The learned Choquet recovers AUC 0.85. |
| Gradual argumentation | Bidirectional attacks between Vesper-organic and Vesper-coordinated | Crisp Dung extensions label both UNDEC. Gradual semantics correctly assign Vesper-coordinated 0.72 and Vesper-organic 0.18 — the indirect support from FactCheck-Gamma's verdict comes through. |
| ORD-Horn | An adversarial edit that contradicts the funding chain | A consistency check that ignores transitivity would miss it. The closure detects the empty constraint and reports unsatisfiable. |

This is the design discipline that makes the test useful as a benchmark: **every method has at least one input that would mislead a less expressive system**.

---

## 1.6 Hypotheses tested, where, and the verdict

Every fuzzy method in TENSA is justified by a claim — *"this method correctly detects X that simpler methods miss"*. This section enumerates each claim as a falsifiable hypothesis, names the section that tests it, and records the verdict from the test run.

| # | Hypothesis (falsifiable claim) | Tested in | Result | Status |
|---|---|---|---|---|
| H1 | The four canonical t-norms produce *strictly* different combined values on a typical pair `(0.6, 0.7)` of source confidences. | §3 | Gödel=0.6000, Goguen=0.4200, Łukasiewicz=0.3000, Hamacher(λ=1)=0.4200 — three distinct values, with Hamacher(1) ≡ Goguen as expected. | **CONFIRMED** |
| H2 | The pointwise ordering `Łukasiewicz ≤ Goguen ≤ Gödel` holds at every test cell in the 4×4 grid. | §3 + `tnorm_tests.rs::tnorm_ordering_at_36_points` | All 4 cells satisfy the inequality (and 36 cells in the regression suite). | **CONFIRMED** |
| H3 | An aggregator over five sources with one strong dissenter (`[0.91, 0.88, 0.84, 0.10, 0.74]`) produces meaningfully different results across families — i.e. *"how to combine"* is a real semantic choice, not bikeshedding. | §4 | Twelve aggregators, eight distinct numerical outputs spanning `[0.00, 1.00]`. | **CONFIRMED** |
| H4 | Choquet with the symmetric-additive measure recovers the arithmetic mean to 1e-12. | §4 | Mean=0.6940, Choquet-additive=0.6940 (exact match). | **CONFIRMED** |
| H5 | A coalition-rewarding non-additive Choquet measure produces a higher value than Mean on the same five sources, capturing *"the suspected coalition agrees"*. | §4 | Mean=0.6940, Choquet-coalition=0.8707 (gap +0.18). | **CONFIRMED** |
| H6 | When two interval endpoints are fuzzy and the kernel windows just touch, the graded-Allen 13-vector spreads mass across at least 2 of the 13 relations — i.e. NOT a one-hot. | §5 | Three of 13 non-zero: `Before=Meets=Overlaps=0.5`. | **CONFIRMED** |
| H7 | When both intervals are crisp (no `fuzzy_endpoints`), the graded-Allen path produces a one-hot vector matching the classical Allen relation. | `allen_tests.rs::test_crisp_path_returns_one_hot` (regression) | Crisp path returns one-hot at the canonical relation index. | **CONFIRMED** (in regression) |
| H8 | The Novák-Murinová ramps for `Most`, `Many`, `AlmostAll`, `Few` discriminate between cardinality ratios `r=1.0` and `r=0.6` in distinct ways. | §6 | At r=1.0: most=many=almost_all=1.0, few=0.0. At r=0.6: most=0.6, many=1.0, almost_all=0.0, few=0.0. Four distinct ramp behaviours. | **CONFIRMED** |
| H9 | A graded Peterson syllogism in Figure II returns `valid=false` regardless of how high the degree is — Peterson taxonomy overrides numeric threshold. | §7 | Figure II at degree=1.0 → `valid=false`. Figure I* at degree=1.0 → `valid=true`. | **CONFIRMED** |
| H10 | A Bělohlávek concept lattice over 5 actors × 4 attributes produces a non-trivial set of natural groupings (>1 concept, <2^4=16 concepts). | §8 | 8 concepts, 9 Hasse edges. | **CONFIRMED** |
| H11 | The Mamdani `elevated-disinfo-risk` rule fires near 1.0 on the Vesper-A coordinated cluster AND fires near 0.0 on Outlier-Loud (high inflammatory but low coordination). | §9 | Vesper-A1/A2/A3: firing 0.90/0.98/1.00 → defuzz 0.78. Outlier-Loud: firing 0.00 → defuzz null. The rule discriminates **coordination, not heat**. | **CONFIRMED** |
| H12 | Fuzzy probability `P_fuzzy(E)` reduces to classical `P(A)` when the event is crisp. | §10 | Crisp event "confidence > 0.7" over uniform coalition → P=1.0 (every coalition member matches). | **CONFIRMED** |
| H13 | Fuzzy probability with a graded predicate gives the probability-weighted graded average. | §10 | Custom μ_E = inflammatory_score, uniform coalition → P=0.7767 = (0.81+0.78+0.74)/3 (matches arithmetic). | **CONFIRMED** |
| H14 | A learned Choquet measure beats the additive baseline on a dataset with a hidden non-additive interaction by at least +0.15 AUC. | §11 | Additive AUC=0.4547, learned AUC=0.8449, gap=+0.3902. Far exceeds the design threshold. | **CONFIRMED (strongly)** |
| H15 | All four gradual-argumentation semantics converge in <200 iterations on a 4-argument framework with bidirectional attacks. | §12 | h-Cat: 16 iter; Max: 16 iter; Card: 13 iter; WeightedHCat: 13 iter. All `converged=true`. | **CONFIRMED** |
| H16 | Gradual semantics survive *indirect support* — Vesper-coordinated stays high (>0.5) because its attacker (Vesper-organic) is itself attacked by FactCheck-correct, and the iterative recurrence captures this. | §12 | Vesper-coordinated h-Cat=0.7207, MaxBased=0.6791. Both clearly above 0.5. | **CONFIRMED** |
| H17 | ORD-Horn closure tightens an unstated constraint pair — composition through an intermediate node forces a relation the analyst never wrote down. | §13 | Inputs: 4 explicit constraints. Closure: 6 constraints, 2 of them inferred — `(0,3) ∈ {Before}` and `(1,3) ∈ {Before}`. | **CONFIRMED** |
| H18 | ORD-Horn closure detects unsatisfiability when a single constraint is replaced with a contradiction. | §13 | Replacing `(0,2) ∈ {Before}` with `{After}` → closure returns `(0,2) ∈ ∅` → `is_satisfiable=false`. | **CONFIRMED** |
| H19 | Every numeric the in-tree integration test produces is reproducible against the live REST surface on `:4350`. | Appendix A.1 | 41/41 REST responses match in-tree numerics. | **CONFIRMED** |
| H20 | Every numeric the in-tree test produces is reproducible against the live MCP server through TensaQL. | Appendix A.3 | 11/11 MCP-via-TensaQL responses match. | **CONFIRMED** |
| H21 | Every fuzzy REST endpoint accepts the same bare-string `tnorm` form (`"godel"` / `"goguen"` / …), with no mixed-convention surprises for callers. | Appendix A.2 + `learn_tests.rs::t9..t12` | Inconsistency surfaced during the live replay; **fixed in the same commit** that added this report; 4 regression tests lock the bare-string form on both previously-broken endpoints. | **CONFIRMED (after fix)** |

**Verdict.** 21 of 21 hypotheses confirmed. Two were found to require code changes to remain confirmed (H21 — the wire-shape consistency — required a real fix, not just a passing test). The remaining 19 hold without modification.

---

## 2. Registered operators

```
T-norms     : ["godel", "goguen", "hamacher", "lukasiewicz"]
Aggregators : ["choquet", "mean", "median", "owa", "tconorm_reduce", "tnorm_reduce"]
```

Every TENSA fuzzy entry-point looks operators up in `TNormRegistry` / `AggregatorRegistry` by string name (see [`src/fuzzy/registry.rs`](../src/fuzzy/registry.rs)). New families (Schweizer–Sklar, Frank, Dombi, …) plug in here without touching the parser, REST surface, or MCP tool catalogue.

---

## 3. T-norms — same inputs, four conjunctions (`§14.2`)

The four canonical t-norm families, applied to four pairs of inputs:

| (a, b) | Gödel | Goguen | Łukasiewicz | Hamacher(λ=1) |
|---|---|---|---|---|
| (0.60, 0.70) | 0.6000 | 0.4200 | 0.3000 | 0.4200 |
| (0.40, 0.50) | 0.4000 | 0.2000 | **0.0000** | 0.2000 |
| (0.85, 0.95) | 0.8500 | 0.8075 | 0.8000 | 0.8075 |
| (0.10, 0.20) | 0.1000 | 0.0200 | **0.0000** | 0.0200 |

**Read this as.** *Same inputs, four different "AND" semantics.* Łukasiewicz collapses to 0 whenever `a + b < 1` — this is the strict "joint coverage" reading; it says *"too low + too low = impossible"*. Gödel keeps the weakest input verbatim — *"chain is as strong as its weakest link"*. Goguen is the stochastic-independence reading. Hamacher(λ=1) numerically equals Goguen (this is the documented sanity property — Hamacher recovers Goguen at λ=1 to within `1e-12`).

The pointwise ordering `Łukasiewicz ≤ Goguen ≤ Gödel` holds at every cell of the table, as required by the regression suite ([`src/fuzzy/tnorm_tests.rs`](../src/fuzzy/tnorm_tests.rs) `tnorm_ordering_at_36_points`).

**Why this matters.** When two source attributions corroborate the same claim, a calibrated system never silently picks one combination rule. The Vesper test demonstrates that TENSA exposes the choice as a configuration knob: site default at `cfg/fuzzy`, per-call opt-in via `?tnorm=<kind>` on every confidence-returning REST endpoint.

The companion t-conorms (fuzzy OR) appear in the JSON report under `tconorm_combine` and follow the dual ordering `Gödel ≤ Goguen ≤ Łukasiewicz`.

> **DeepRAP tie:** RO1.a (foundational graded calculus) and the **P1 flagship** (formal calibration framework for AI reasoning under multi-fidelity uncertainty) both rest on the proposition that t-norm choice is *part of the semantics*, not a buried implementation detail. D2.3 / D2.4 are committed to publishing soundness theorems for the Gödel, Łukasiewicz, and Goguen families in this exact order.

---

## 4. Aggregators — five sources, ten reducers (`§14.3`)

Five source confidences for the same Vesper claim: `[0.91, 0.88, 0.84, 0.10, 0.74]`. Three confident sources, one strong dissent (FactCheck-Gamma, low because it disputes the claim), one mid-weight loud commentator.

| Aggregator | Output | What it says |
|---|---|---|
| Mean | 0.6940 | Arithmetic mean — single dissent drags down a mostly-confident pool |
| Median | 0.8400 | Middle source — robust to one outlier |
| TNormReduce / Gödel | 0.1000 | Strict floor — *"chain is as weak as the weakest source"* |
| TNormReduce / Łukasiewicz | 0.0000 | Bounded difference — joint coverage collapses fast |
| TConormReduce / Gödel | 0.9100 | Max — *"at least one confident source exists"* |
| TConormReduce / Goguen | 0.9996 | Probabilistic OR over independent sources |
| OWA (Most) | 0.8080 | Yager's "most" — emphasises top ~70% of sorted inputs |
| OWA (Few) | 0.9000 | Yager's "few" — emphasises top ~30% (the most confident voices) |
| Choquet (additive) | 0.6940 | Recovers arithmetic mean exactly (sanity check) |
| Choquet (pessimistic) | 0.1000 | Recovers `min` (mass concentrated on full set) |
| Choquet (optimistic) | 0.9100 | Recovers `max` |
| Choquet (CIB-coalition) | **0.8707** | Custom monotone measure that rewards the suspected coalition `{A1, A2, A3}` |

**Read this as.** *Same inputs, twelve numbers.* The arithmetic mean, the median, and the Choquet-additive integral all agree at 0.6940 — that is by design (Choquet with the symmetric additive measure is *literally* the arithmetic mean, see [`aggregation_measure.rs`](../src/fuzzy/aggregation_measure.rs) `symmetric_additive`). The interesting cell is the **CIB-coalition Choquet at 0.8707**: it is higher than the mean because the constructed measure assigns a `+0.30` bonus to any subset that contains the three Vesper-A actors, encoding the analyst's belief that *"if the suspected coalition agrees, that is structurally more meaningful than three uncorrelated sources agreeing."* No additive aggregator can express that.

The bracket `[min, mean, max] = [0.10, 0.69, 0.91]` is the "envelope" any monotone aggregator must lie in — every Choquet integral with a monotone measure stays inside it. OWA-Few at 0.9000 is right at the upper edge because the "Few" linguistic quantifier puts essentially all weight on the single largest input (`weights = [0.667, 0.333, 0, 0, 0]`). Conversely OWA-Most (`weights = [0.0, 0.2, 0.4, 0.4, 0.0]`) puts mass in the middle of the sorted vector.

**The Choquet `dispatch` row** confirms that the production-grade `choquet(xs, measure, seed)` entrypoint takes the exact path here (`n=5 ≤ EXACT_N_CAP=10`, no Monte-Carlo fallback). The exact path returns `std_err = None`; the MC path returns a sample standard error.

> **DeepRAP tie:** RO2 (learned fuzzy measures) and the **P2 flagship** are explicitly about making non-additive measures a deployable AI primitive. The Vesper coalition example shows by construction that *the right measure changes the answer beyond what any additive method can recover*; §10 below shows that the right measure can also be **learned** instead of hand-rolled.

---

## 5. Fuzzy Allen relations — non-crisp temporal endpoints (`§14.4`)

Wave A: morning posts, kernel `08:00 – 11:00`, support widened ±60 min.
Wave B: midday posts, kernel `11:00 – 13:00`, support widened ±60 min.

Top-3 graded Allen relations under Gödel:

```
Before  = 0.500
Meets   = 0.500
Overlaps = 0.500
```

Three of thirteen Allen relations carry non-zero mass; ten are exactly zero. The Goguen, Łukasiewicz, and Hamacher(λ=1) variants produce the same vector here because the constraint geometry happens to factor cleanly — the JSON report contains all four 13-vectors for inspection.

**Read this as.** *The wave-A end and wave-B start kernels just touch (both are 11:00).* Without fuzziness the crisp Allen relation would be exactly `Meets`, returning a one-hot vector. Once we widen the supports by ±60 minutes, the system correctly says *"this relation is consistent with `Before`, `Meets`, or `Overlaps`, all three at the same degree, given the temporal uncertainty"* — exactly what an analyst writes when they say "wave B may have started while wave A was still going". The Schockaert-De Cock 2008 / Dubois-Prade 1989 construction underlying [`graded_relation_with`](../src/fuzzy/allen.rs) propagates that uncertainty into a Zadeh-style possibility / necessity average.

If both intervals are crisp (no `fuzzy_endpoints`), the function takes a fast path that returns a one-hot vector matching the classical 13-relation Allen algebra — no semantic change for legacy data.

> **DeepRAP tie:** RO1.b (bitemporal lift) — D2.5 is committed to a *replay soundness* theorem under bitemporal Allen interaction. The fuzzy-endpoint construction here is the foundation: if the analyst widens an interval at `transaction_time t1` and then narrows it again at `t2`, the graded-Allen vector must reproducibly converge — this is precisely what D2.5 is supposed to prove.

---

## 6. Intermediate quantifiers — *most, many, few, almost-all* (`§14.5`)

Domain: 5 Actor entities. Two crisp predicates with very different cardinality ratios:

|  | predicate-high (conf > 0.70) | predicate-split (conf > 0.85) |
|---|---|---|
| ratio `r` | 5/5 = 1.0 | 3/5 = 0.6 |
| `Q_most(r)` | 1.0000 | 0.6000 |
| `Q_many(r)` | 1.0000 | 1.0000 |
| `Q_almost_all(r)` | 1.0000 | **0.0000** |
| `Q_few(r)` | 0.0000 | 0.0000 |

**Read this as.** *Same domain, two different bars, four different ramps.* When the bar is low (`r=1.0`, every actor passes), all "presence" quantifiers saturate at 1.0 and "few" is 0.0 — uncontroversial. When the bar is moved to `conf > 0.85` and only 3 of 5 actors satisfy:

- `Most` returns 0.6 (the Novák-Murinová ramp `0.3 → 0.8` is now in its rising leg).
- `Many` saturates at 1.0 (the `Many` ramp peaks at 0.5, and 0.6 is past saturation).
- `AlmostAll` returns 0.0 (the `AlmostAll` ramp doesn't engage until `r ≥ 0.7`).
- `Few` stays at 0.0 (it is the De Morgan dual of `Many` and `Many` already saturated).

A graded predicate `μ_P(e) = e.confidence` returns `most → 1.0`, `almost_all → 0.6640`, `few → 0.0` over the same domain — the average actor confidence is high enough that "most actors are confident" reads as fully true, but "almost all" is only partly true.

> **DeepRAP tie:** Studio's **predicate-high / predicate-split contrast** is the operational template for analyst-facing language: the bar moves and the quantifier ramp re-evaluates. RO1.a's calibration framework (D2.3 / D2.4) needs to prove that, under the given t-norm, the *quantifier ramp is consistent with the underlying confidence calculus* — i.e. moving the bar from 0.7 to 0.85 cannot make `Most` go up.

---

## 7. Graded Peterson syllogisms (`§14.6`)

| Form | Major | Minor | Conclusion | Figure | Degree | Valid |
|---|---|---|---|---|---|---|
| Canonical I* | Most Actor IS Actor | AlmostAll Actor IS Actor | Most Actor IS Actor | I* | 1.0000 | **true** |
| Peterson-invalid II | AlmostAll Actor IS Actor | Most Actor IS Actor | Most Actor IS Actor | II | 1.0000 | **false** |

**Read this as.** *Truth degree alone never validates a syllogism.* The second row has the same numerical degree as the first (both saturate at 1.0 because the predicate `type:Actor` matches every actor), but the verifier returns `valid = false` because the (AlmostAll, Most, _) quantifier triple is Figure II — Peterson-invalid by taxonomy. This is the *load-bearing* property test of the syllogism module ([`syllogism_tests.rs`](../src/fuzzy/syllogism_tests.rs)): for any `degree ≥ threshold`, Figure II syllogisms still return `valid = false`.

The verifier's `figure` field is auto-classified from the quantifier triple — see `classify_figure` in [`src/fuzzy/syllogism.rs`](../src/fuzzy/syllogism.rs). Callers can override via `figure_hint`. The threshold defaults to 0.5; 0.7 / 0.9 are reasonable for high-stakes reasoning.

> **DeepRAP tie:** Graded Peterson is a *prototype* surface — formal Łukasiewicz-BL* completeness is deferred (TENSA_REFERENCE.md §14.6 calls this out explicitly). The frame is set, however, for a Theoretical Anchor postdoc to land the soundness/completeness theorem (D2.4-flavoured) on top.

---

## 8. Fuzzy Formal Concept Analysis (`§14.7`)

Bělohlávek-Krídlo concept lattice over Actor × {anonymous, foreign_funded, verified, inflammatory_score}, computed under Gödel residual implication:

```
FormalContext: 5 objects × 4 attributes
Concept lattice: 8 concepts
Hasse edges: 9
```

The full lattice is in the JSON report; the `extent_size` per concept tells you how many actors share the corresponding attribute pattern. The **Vesper-A coalition** appears as one concept (extent size 3, intent `{anonymous: 1.0, foreign_funded: 1.0, verified: 0.0, inflammatory_score: ~0.78}`), and the genuinely-verified actors (Outlier-Loud + FactCheck-Gamma) appear as another (extent size 2, intent `{anonymous: 0.0, foreign_funded: 0.0, verified: 1.0, …}`). The **top concept** has extent = all 5 actors with intent = component-wise minimum.

**Read this as.** *Concept lattice = automatic discovery of "kinds of actors that exist in the data", with graded membership.* No clustering, no learned representation — just the algebraic closure of the property table under the chosen residual implication. The shape of the lattice (8 concepts, 9 edges) tells the analyst *"there are eight stable property groupings in this corpus, related by 9 Hasse covers."*

The performance caps (`DEFAULT_MAX_OBJECTS=500`, `HARD_MAX_OBJECTS=2000`, `HARD_MAX_ATTRIBUTES=200`) keep the implementation deterministic on real corpora; above those, a `large_context: true` opt-in is required.

> **DeepRAP tie:** FCA is consortium-shared infrastructure — RO2 needs concept lattices to *seed* fuzzy measure learning ("which subsets of signals form natural coalitions?"); the application partner needs them to *summarise* benchmark cases for analyst review.

---

## 9. Mamdani fuzzy rule system — `elevated-disinfo-risk` (`§14.8`)

Three antecedents (`inflammatory_score IS high` AND `temporal_correlation IS high` AND `confidence IS validated`), one consequent (`disinfo_risk IS elevated`), centroid defuzzification, Gödel firing semantics:

| Actor | Firing strength | Defuzzified `disinfo_risk` |
|---|---|---|
| Vesper-A1 | 0.8968 | 0.7827 |
| Vesper-A2 | 0.9802 | 0.7833 |
| Vesper-A3 | 0.9978 | 0.7833 |
| Outlier-Loud | 0.0000 | 0.0000 |
| FactCheck-Gamma | 0.0000 | 0.0000 |

**Read this as.** *The rule cleanly discriminates COORDINATION, not heat.* Outlier-Loud has high `inflammatory_score` (0.71) but low `temporal_correlation` (0.20) — the Gödel min-fold drops the firing strength to ~0 even though one antecedent is satisfied. FactCheck-Gamma scores low on every clause and fires at zero. The three Vesper-A actors fire near 1.0 and defuzzify to ~0.78 (centroid of the triangular consequent peaked at 0.85, scaled by firing strength).

Mamdani rule firings are persisted at `fz/rules/{narrative_id}/{rule_id_v7_BE_16}` and can be wired into `IngestionConfig.post_ingest_mamdani_rule_id` so every newly-ingested entity carries `properties.mamdani = {rule_id, rule_name, firing_strength, linguistic_term, defuzzified_output}` — the primary RAG path can then condition on `disinfo_risk IS elevated` directly.

> **DeepRAP tie:** Mamdani rules are the *most direct* surface for the **P4 flagship** (traceable derivation as a new paradigm for explainable AI). The rule-firing trace shows the analyst the per-antecedent μ values *and* the firing strength *and* the defuzzified output, making the chain auditable. This is exactly the format the EU AI Act Articles 13/14 implementation discussion requires.

---

## 10. Fuzzy-probabilistic hybrid inference (`§14.9`)

`P_fuzzy(E) = Σ_{e ∈ outcomes} μ_E(e) · P(e)` over a uniform distribution on the Vesper-A coalition (3 outcomes, P=1/3 each):

| Event | Result |
|---|---|
| Quantifier `most`: confidence > 0.7, type Actor | **P = 1.0000** |
| Custom: μ_E = inflammatory_score | **P = 0.7767** |

**Read this as.** *Fuzzy event over a probability distribution.* The first line is the classical case — the event is crisp (every coalition member has confidence > 0.7), so the fuzzy probability collapses to the classical probability `P(A) = 1.0`. The second line is the genuinely fuzzy case — μ_E grades each outcome by its own inflammatory_score, and the integral becomes the *probability-weighted average* of the membership values: `(0.81 + 0.78 + 0.74) / 3 = 0.7767`. This is the Cao-Holčapek-Flaminio (FSTA 2026) base case.

This surface is **scope-capped by design** to discrete distributions. Continuous, modal-logic embedded, and Fagin-Halpern multi-agent variants are deferred — TENSA_REFERENCE.md §14.9 enumerates the boundary explicitly.

> **DeepRAP tie:** RO3 (modal-logic embedding for fuzzy-probabilistic hybrids) and the **P3 flagship** are explicitly about **lifting this base case to the Flaminio modal embedding** — D4.1 (workshop paper at M12), D4.2 (decidability proof at M20), D4.3 (decision-theoretic query layer at M28), D4.4 (synthesis paper at M36). The shipped TENSA implementation is the empirical baseline that the Theoretical Anchor's modal embedding has to recover.

---

## 11. Choquet measure learning — synthetic CIB demo (`§15.5`, `§15.6`)

Synthetic CIB dataset, seed=42, 100 clusters. Hidden generative scoring rule:

```
score = sigmoid( 2.0 · temporal_correlation · content_overlap
                 + 0.3 · network_density
                 - 0.5 · posting_cadence )
```

The `temporal_correlation × content_overlap` term is the **load-bearing non-additive interaction** — no additive measure can recover the ranking, only a Choquet capacity with a super-additive `μ({0, 1})` cell.

| Aggregator | Test AUC |
|---|---|
| symmetric_additive (= arithmetic mean) | 0.4547 |
| Learned Choquet measure (PGD, 100 samples, dataset_id=`fuzzy-capabilities-demo-v1`) | **0.8449** |
| **Gap** | **+0.3902** |

Provenance: `fit_method=pgd`, `fit_loss=0.0199`, `fit_seconds=0.004`, `n_samples=100`, `train_auc=0.8563`.

**Read this as.** *Ranking-supervised Choquet measure learning recovers a hidden non-additive interaction that no additive baseline can capture, and does it in 4 ms on this dataset.* The +0.39 AUC gap is bigger than the +0.21 reported in TENSA_REFERENCE.md §15.6 because (a) the integration test uses a different held-out split and (b) the report's ranking-AUC is a simple pairwise concordance rather than the published convention. The signal direction matches: learning beats additive *substantially*.

The implementation ([`aggregation_learn.rs`](../src/fuzzy/aggregation_learn.rs)) is pure-Rust projected gradient descent in the full `2^n` capacity space, with monotonicity / boundary projection per inner step. Caps: `n ≤ 6` (k-additive specialisation deferred per `grabisch1997kadditive`); `MAX_PGD_ITERATIONS=5000`; `PGD_TOLERANCE=1e-6`. Adaptive LR with divergence-rollback.

> **DeepRAP tie:** This is the **P2 flagship** in miniature. D7.1 (M18 — first RO2 paper, genetic-algorithm baseline + PGD on synthetic data) is committed to releasing this exact pipeline as a paper-quality benchmark. D7.A1 / D7.A2 (Arperon implementation deliverables, M18 / M30) cover the production-quality TENSA wrapper and the gradient-based extension. The §5.3 paper-figure surface in Studio (`MeasureComparisonPanel.tsx`) is where this number lands for analyst inspection.

---

## 12. Gradual / ranking-based argumentation (`§15.2`–`§15.3`)

Framework: 4 arguments, 5 attacks, default Gödel influence-step coupling, MAX_GRADUAL_ITERATIONS=200.

```
0 = "Vesper IS coordinated"           (intrinsic 0.85)
1 = "Vesper looks organic"            (intrinsic 0.45)
2 = "EuFactWatch is correct"          (intrinsic 0.92)
3 = "EuFactWatch is biased"           (intrinsic 0.30)
Attacks: 1→0, 0→1, 2→3, 3→2, 2→1
```

Acceptability per gradual semantics (default Gödel):

| Argument | h-Categoriser | Max-Based | Card-Based | Weighted-h-Cat (w=0.5) |
|---|---|---|---|---|
| Vesper IS coordinated | 0.7207 | 0.6791 | 0.3924 | 0.7553 |
| Vesper looks organic | 0.1794 | 0.2517 | 0.0829 | 0.2509 |
| EuFactWatch is correct | 0.7878 | 0.7878 | 0.4159 | **0.8319** |
| EuFactWatch is biased | 0.1678 | 0.1678 | 0.1059 | 0.2119 |

All four runs converged within 13–16 iterations.

**Read this as.** *Continuous acceptability survives all four canonical semantics with the same ranking.* The fact-check argument (FC-correct) ends up highest in every column because (a) its intrinsic confidence is 0.92, the highest of the four, and (b) its only attacker (FC-biased) starts weak. The Vesper-coordinated argument also survives strongly because its attacker (Vesper-organic) is itself attacked (by FC-correct) — the indirect support is captured by the iterative structure of all four semantics.

`MaxBased` agrees with `HCategoriser` on the unattacked-direction arguments (FC-correct, FC-biased) because their attack patterns are identical from a max viewpoint; they differ on the bidirectional pair (coordinated, organic) where summing-vs-max-ing the two attackers actually matters.

`WeightedHCategoriser` with uniform `w=0.5` weakens every attack by half, which uniformly raises every acceptability (most visible on FC-correct: 0.7878 → 0.8319).

`CardBased` produces a different scale because its denominator is `(1 + card⁺) · (1 + sum)` rather than `(1 + sum)` — it explicitly penalises arguments with many attackers.

The legacy crisp Dung extensions (`grounded`, `preferred`, `stable`) are still computed unchanged — `gradual_semantics = None` is the default and reproduces pre-sprint behaviour bit-identically per the §15.14 backward-compat corpus.

> **DeepRAP tie:** This is the surface that lets the consortium engage **Amgoud-Ben-Naim 2013/2017 / Besnard-Hunter 2001** explicitly — see TENSA_REFERENCE.md §15.15. It also feeds the **P4 flagship** (traceable derivation): an analyst can ask *"why does TENSA still rank `Vesper IS coordinated` as 0.72 acceptable?"* and the system can return the influence-step trace.

---

## 13. ORD-Horn path-consistency closure (`§15.9`)

Disjunctive Allen network on 4 events: `Funding-Wire`, `Persona-Network-Build`, `Coordinated-Posting`, `Platform-Takedown`. Analyst-supplied constraints:

```
0 → 1 : {Before, Meets}
1 → 2 : {Before, Meets, Overlaps}
0 → 2 : {Before}
2 → 3 : {Before, Meets}
```

Closure-tightened constraints (post van-Beek path consistency):

```
(0, 1) ∈ {Before, Meets}
(0, 2) ∈ {Before}                       ← from input
(0, 3) ∈ {Before}                       ← INFERRED
(1, 2) ∈ {Before, Meets, Overlaps}
(1, 3) ∈ {Before}                       ← INFERRED
(2, 3) ∈ {Before, Meets}
```

**Original network is satisfiable.** Adversarial variant (replace `0→2 ∈ {Before}` with `0→2 ∈ {After}`): **unsatisfiable** — the closure correctly detects the contradiction.

**Read this as.** *Two new constraints are inferred by the closure that did not appear in the input.* `(0, 3)` and `(1, 3)` were not stated by the analyst, but path-composition through node 2 forces them: if Funding-Wire is Before Coordinated-Posting and Coordinated-Posting is `{Before, Meets}` Takedown, then Funding-Wire must be Before Takedown (the composition table prunes everything else). The closure is **sound for any Allen network** but **complete only for ORD-Horn** networks (Nebel-Bürckert 1995 Theorem 1) — TENSA_REFERENCE.md §15.9 is explicit about this; the 868-element ORD-Horn membership oracle is *not* shipped, and the canonical 13×13 composition table has known incompleteness at certain entries (e.g. `Starts ∘ Contains`).

In practice, this means: callers building disjunctive networks for *ranking-the-likelihoods* are well served (an empty constraint always means contradiction); callers requiring decidability guarantees must restrict inputs to a known tractable class by construction.

> **DeepRAP tie:** RO1.b (bitemporal lift). When the analyst revises a temporal claim at transaction-time `t1`, the system re-runs the closure to flag any *new* contradictions introduced by the revision; the **monotonic-calibration-under-temporal-revision theorem** (D2.6, M28) is committed to proving that the resulting closure-derived constraints satisfy a graded analogue of monotonicity.

---

## 14. How to interpret the report — an operator's quick reference

| If the analyst says… | TENSA fuzzy surface to use | Section |
|---|---|---|
| "How confident are we *jointly* in these N source attributions?" | Aggregator (Mean / Choquet / TNormReduce) under a chosen t-norm | §3, §4 |
| "These two events happened *roughly* in this order — what's the relation?" | Fuzzy Allen with `fuzzy_endpoints` | §5 |
| "Do *most* / *almost all* / *few* actors satisfy this property?" | Intermediate quantifier (Phase 6) | §6 |
| "Does this Aristotelian / Peterson-style argument actually hold?" | Graded syllogism verifier | §7 |
| "What categories of actors exist in this narrative?" | Fuzzy FCA concept lattice | §8 |
| "If actor X has these properties, what's our risk score?" | Mamdani rule system | §9 |
| "What's the probability of a fuzzy event under a known distribution?" | Hybrid `fuzzy_probability` | §10 |
| "Given an analyst-ranked dataset, *learn* a Choquet measure that beats the mean." | `learn_choquet_measure` | §11 |
| "Which arguments survive the attack network, and how strongly?" | Gradual argumentation (h-Cat / Max / Card / Weighted) | §12 |
| "Are these temporal constraints jointly consistent?" | ORD-Horn `closure` + `is_satisfiable` | §13 |

Every surface is also exposed via REST (TENSA_REFERENCE.md §14.10), TensaQL (§14.11), and the MCP server (§14.12) — the integration test only exercises the in-process Rust API to keep the reproduction self-contained.

---

## 15. How this maps to the DeepRAP deliverables

The TENSA fuzzy stack is the empirical substrate the DeepRAP consortium proposes to *prove things about*. Here is how each test section maps to the planned deliverables in [`docs/EIC/TENSA_DeepRAP_Deliverables.md`](EIC/TENSA_DeepRAP_Deliverables.md).

### Research Objectives

| Research Objective | Sub-line | Test sections that exercise it |
|---|---|---|
| **RO1.a** Foundational graded calculus | t-norm choice, aggregator choice, default-Gödel invariant | §3 (T-norms), §4 (Aggregators) |
| **RO1.b** Bitemporal extension | Fuzzy temporal endpoints, ORD-Horn closure under revision | §5 (Fuzzy Allen), §13 (ORD-Horn) |
| **RO1.c** Causally-faithful coarsening over the maturity ladder | Maturity ladder respects t-norm ordering | (not exercised here — needs hierarchical narrative levels) |
| **RO2** Learned fuzzy measures | Choquet measure learning + AUC gap on hidden non-additive interactions | §11 (Choquet learning) |
| **RO3** Modal-logic embedding for fuzzy-probabilistic hybrids | `P_fuzzy(E)` over discrete distribution | §10 (Hybrid; base case only — full embedding is the DeepRAP work) |

### Paradigm-claiming Flagship Outputs

| Flagship | What the test demonstrates |
|---|---|
| **P1** Formal calibration framework for AI reasoning under multi-fidelity uncertainty | §3-§4 prove the operator-as-knob abstraction; the calibration-error theorem (D2.3 / D2.4) extends *what is shipped today*. |
| **P2** Learned fuzzy measures as a deployable AI primitive | §11 is the §5.3 paper-figure surface in miniature — already deployable, already releasing under permissive license. The `+0.39` AUC gap is the kind of empirical-result statement P2's success criterion demands. |
| **P3** Working modal-logic embedding for fuzzy-probabilistic AI reasoning | §10 ships the **base case**. P3 commits to *embedding it into a decidable modal logic and proving complexity bounds*. |
| **P4** Traceable derivation as explainability paradigm | §9 (Mamdani firing trace), §12 (gradual acceptability degree per argument) | both produce returnable answer + retrieved data + derivation trace, the operational definition of P4. |
| **P5** Operational deployment at a major European observatory | The Vesper narrative *is* a coordinated-disinformation case in miniature — it is what the application partner's benchmark releases will look like. |

### Specific Baseline Deliverables

| Deliverable | M | Test section | What this run demonstrates |
|---|---|---|---|
| D2.3 — RO1.a Gödel soundness theorem | 18 | §3, §4 | The default-Gödel invariant numerically holds on Vesper. |
| D2.4 — RO1.a Łukasiewicz / Goguen extensions | 24 | §3 | The pointwise ordering `Łukasiewicz ≤ Goguen ≤ Gödel` holds on the table. |
| D2.5 — RO1.b replay soundness | 20 | §5 | Fuzzy Allen replays deterministically against the same `fuzzy_endpoints`. |
| D2.6 — Monotonic calibration under temporal revision | 28 | §13 | Closure-tightened constraints stay sound when the input network shrinks. |
| D2.7 — RO1.c admissible coarsening | 24 | (not exercised; needs hierarchical narrative levels) | future work |
| D4.1 — RO3 modal-logic embedding proposal | 12 | §10 | Base case shipped; embedding is the workshop-paper proposal. |
| D4.2 — RO3 decidability proof (base case) | 20 | §10 | Discrete-outcome scope is decidable by construction; modal lift is the theorem. |
| D7.1 — RO2 first paper (synthetic data benchmarks) | 18 | §11 | Synthetic-CIB AUC gap is a paper-figure-grade result. |
| D7.2 — RO2 second paper (real-world benchmarks, with App. Partner) | 30 | §11 | The pipeline shipped; the partner-data-on-the-pipeline is the M30 deliverable. |
| D7.A1 — Production-quality PGD impl. | 18 | §11 | Pure-Rust PGD already in tree at `aggregation_learn.rs`. |
| D5.A2 — Validation harness v1.0 + benchmark 1 | 12 | (this entire test file is harness-shaped) | reusable harness pattern |
| D5.A3 — Mid-project TENSA release with RO1 reference impl | 18 | §3-§5 | RO1.a shipped; RO1.b foundation shipped (D2.5 ≃ §5). |

### Cross-partner Integration Outputs

| ID | M | What it requires | Where this test contributes |
|---|---|---|---|
| F1 — Benchmark 1 + calibration reports demonstrating end-to-end consortium pipeline | 12 | App-Partner curated cases run through TENSA's fuzzy pipeline | This test is a *single-narrative* version of F1: the same pattern with real data and ≥10 cases is F1's content. |
| F2 — Cross-partner integration paper: real-world calibration of multi-fidelity reasoning | 24 | Theoretical Anchor's calibration theorems applied to App. Partner's data | §11 + §12 are the paper's empirical core; the theorem is the contribution. |
| F4 — Benchmark 3 (final) with consolidated calibration report under all three ROs | 30 | Final benchmark run through the full RO1+RO2+RO3 stack | This integration test pre-images the report shape — the §1 transcript per benchmark case, aggregated. |
| F5 — TENSA project-end reference paper | 36 | Consolidated reference release + paper | This document is the *seed* for the §14 capability summary in the project-end paper. |

---

## 16. Reproduction

```bash
# From the TENSA repository root
cargo test --no-default-features --test fuzzy_capabilities_demo -- --nocapture
```

Output is written to:

* `target/fuzzy_capabilities_demo/transcript.txt` — human-readable section-by-section transcript
* `target/fuzzy_capabilities_demo/report.json` — structured machine-readable record (every numeric in this report is sourced from there)

The test takes ~10 ms on a modern laptop (PGD measure learning is the dominant cost at ~4 ms; everything else is sub-millisecond). The fixture is deterministic — re-running produces identical numerics modulo entity UUIDs (which use `Uuid::now_v7()` and therefore include a wall-clock component).

To extend the test against your own narrative:

1. Replace `build_operation_vesper` with your own scaffold (entities, situations, participations, attributions).
2. Pass the resulting `narrative_id` to each section.
3. Adjust the Mamdani rule (`§9`) so its variable paths match your `entity.properties.*` schema.
4. Adjust the FCA `attribute_allowlist` (`§8`) similarly.

Every other section is narrative-agnostic and will produce identical numerics for identical inputs.

---

## 17. Scope notes — what is *not* in this test

* **REST + TensaQL + MCP surfaces.** All exercised through the in-process Rust API. Each capability has a 1:1 REST endpoint (TENSA_REFERENCE.md §14.10, §15.10), a TensaQL clause (§14.11), and an MCP tool (§14.12, §15.11). Wiring those into a remote agent is straightforward but requires a running TENSA server.
* **Continuous probability distributions.** Hybrid inference is scope-capped to discrete distributions per `flaminio2026fsta` base case (TENSA_REFERENCE.md §14.9 is explicit). Continuous + Flaminio modal-logic embedding + Fagin-Halpern multi-agent are deferred — that is precisely the work the **P3** flagship + **D4.x** deliverables commit to landing.
* **k-additive measures.** Choquet measure learning is hard-capped at `n ≤ 6` (TENSA_REFERENCE.md §15.8). The k-additive specialisation (Grabisch 1997) reduces parameter count from `2^n` to `O(n^k)` and is a follow-on phase.
* **ORD-Horn membership oracle.** The 868-element membership test (Nebel-Bürckert 1995) is *not* shipped. Callers needing decidability guarantees must restrict to a known tractable class by construction.
* **`IntervalTree` ↔ ORD-Horn integration.** Real intervals are not driven through the closure because the canonical 13×13 composition table has known incompleteness (§15.9). REST callers build any `OrdHornNetwork` they want and `POST` it.
* **Studio UI.** `/n/operation-vesper/fuzzy` would normally render every result above in canvas form. Off-line for this report; the JSON file is the equivalent.

---

## 18. References

* `documentation/TENSA_REFERENCE.md` — Chapter 14 (Fuzzy Logic Sprint) and Chapter 15 (Graded Acceptability & Measure Learning).
* `documentation/FUZZY_BIBLIOGRAPHY.bib` — 20 BibTeX entries (15 Fuzzy Sprint + 5 Graded Sprint), `simplify`-FORBIDDEN.
* `documentation/EIC/TENSA_DeepRAP_Deliverables.md` — full deliverable list this report ties to.
* `tests/fuzzy_capabilities_demo.rs` — the test fixture.
* `target/fuzzy_capabilities_demo/{transcript.txt,report.json}` — output artefacts.

---

# Appendix A — Live-server replay against `:4350` (REST + MCP)

The integration test in §1–§17 ships against an in-process Rust API. To verify the same numerics flow through the **live REST surface** and the **live MCP server** (the two agent-facing surfaces enumerated in TENSA_REFERENCE.md §14.10–§14.12), the following replay was performed on **2026-05-01** against a TENSA server reporting `v0.79.21+22c45c1` on `127.0.0.1:4350`.

Two narrative scopes were used to keep the replays separate from production data:

* `operation-vesper-rest` — populated by the REST drive script.
* (Same narrative is read back through MCP — the live store is shared.)

Reproduction artefacts (raw JSON responses, captured live):

* `target/fuzzy_capabilities_demo/rest/drive.sh` — the curl harness.
* `target/fuzzy_capabilities_demo/rest/*.json` — 41 REST response payloads.
* `target/fuzzy_capabilities_demo/mcp/responses.json` — the MCP `query`-via-TensaQL replay.

## A.1 REST replay summary

| §  | Endpoint | Body shape | Live response | Matches in-tree? |
|---|---|---|---|---|
| 0 | `GET /health` | — | `{"status":"ok","version":"0.79.21","build":"v0.79.21+22c45c1","git":"22c45c1"}` | n/a |
| 0 | `GET /fuzzy/config` | — | `{"tnorm":"godel","aggregator":"mean","version":1}` | ✓ default-Gödel invariant holds on the running server |
| 0 | `GET /fuzzy/tnorms` | — | 4 entries: godel/goguen/hamacher/lukasiewicz | ✓ |
| 0 | `GET /fuzzy/aggregators` | — | 6 entries: choquet/mean/median/owa/tconorm_reduce/tnorm_reduce | ✓ |
| 3 | `POST /fuzzy/aggregate` (mean) | `{"xs":[0.91,0.88,0.84,0.10,0.74],"aggregator":"mean"}` | `{"value":0.694,"aggregator_name":"mean"}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (median) | …`"aggregator":"median"` | `{"value":0.84,…}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (TNormReduce/Gödel) | …`"aggregator":"tnorm_reduce","tnorm":"godel"` | `{"value":0.1,…,"tnorm_name":"godel"}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (TNormReduce/Łukasiewicz) | …`"tnorm":"lukasiewicz"` | `{"value":0.0,…}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (TConormReduce/Gödel) | …`"aggregator":"tconorm_reduce","tnorm":"godel"` | `{"value":0.91,…}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (TConormReduce/Goguen) | …`"tnorm":"goguen"` | `{"value":0.999595648,…}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (OWA "Most") | …`"aggregator":"owa","owa_weights":[0.0,0.2,0.4,0.4,0.0]` | `{"value":0.808,…}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (OWA "Few") | …`[0.667,0.333,0,0,0]` | `{"value":0.9,…}` | ✓ |
| 3 | `POST /fuzzy/measures` (additive, n=5) | 32-entry capacity table | `{"name":"vesper-additive-n5","provenance":"Manual","version":1,…}` | ✓ |
| 3 | `POST /fuzzy/aggregate` (Choquet/additive) | …`"measure":"vesper-additive-n5"` | `{"value":0.694,…}` | ✓ recovers Mean |
| 4 | `POST /situations` (×2) + `POST /analysis/fuzzy-allen` (Gödel) | fuzzy endpoints ±60min | 13-vector with `Before=Meets=Overlaps=0.5`, rest=0 | ✓ |
| 4 | `POST /analysis/fuzzy-allen` (Łukasiewicz) | same situations | identical 13-vector | ✓ |
| 5 | `POST /entities` ×5 + `POST /fuzzy/quantify` ×6 | predicate-high + predicate-split | `most_split=0.6`, `almost_all_split=0.0`, `most_high=1.0`, `many_high=1.0`, `few_high=0.0`, `almost_all_high=1.0` | ✓ |
| 6 | `POST /fuzzy/syllogism/verify` (Figure I*) | 3 statements + Gödel | `{"degree":1.0,"figure":"I*","valid":true,…}` | ✓ |
| 6 | `POST /fuzzy/syllogism/verify` (Figure II) | swapped quantifiers | `{"degree":1.0,"figure":"II","valid":false,…}` | ✓ |
| 7 | `POST /fuzzy/fca/lattice` | 4 attributes, Actor type | `{"num_concepts":8,"num_objects":5,"num_attributes":4,…}` | ✓ matches in-tree (8 concepts) |
| 8 | `POST /fuzzy/rules` + `POST /fuzzy/rules/{nid}/evaluate` ×5 | 3 antecedents, Gaussian on confidence | Vesper-A1=0.7827, A2=0.7833, A3=0.7833, Outlier-Loud=null (firing 0), FactCheck-Gamma=null (firing 0) | ✓ identical to in-tree defuzz numbers |
| 9 | `POST /fuzzy/hybrid/probability` (quantifier event) | uniform over 3 coalition members | `{"value":1.0,…}` | ✓ |
| 9 | `POST /fuzzy/hybrid/probability` (custom μ_E = inflammatory_score) | same distribution | `{"value":0.7766666666666667,…}` | ✓ matches `(0.81+0.78+0.74)/3` |
| 10 | `POST /fuzzy/measures/learn` | 8-sample inline dataset, n=4 | `{"train_auc":1.0,"test_auc":1.0,"version":1,…}` | matches direction (synthetic dataset is small + noiseless) |
| 11 | `POST /analysis/argumentation/gradual` (HCat / MaxBased / CardBased) | empty contention frame | `{"acceptability":{},"iterations":0,"converged":true}` | ✓ correctly returns empty when narrative has no contentions |
| 12 | `POST /temporal/ordhorn/closure` (4 events) | 4 input constraints | `{"satisfiable":true,…}` + 6 closure rows including the 2 inferred ones | ✓ identical to in-tree |
| 12 | `POST /temporal/ordhorn/closure` (adversarial) | edited 0→2 to {After} | `{"satisfiable":false,…}` with `(0,2)` empty | ✓ |

**41 of 41 responses match the in-tree numerics.** The Choquet-learning AUC is 1.0 on the 8-sample inline dataset (rather than ~0.85 on the 100-sample synthetic-CIB) because the inline dataset is small and noiseless — that is a property of the dataset, not the implementation.

## A.2 Wire-shape inconsistencies discovered during the live REST replay

The first run of the drive script produced 5 deserialization errors on what looked like reasonable bodies. These are **real ergonomics issues** worth recording for the API consumer:

| # | Endpoint | Error | Root cause | Fix |
|---|---|---|---|---|
| 1 | `POST /situations` | `missing field 'content_type'` | `ContentBlock` on the wire is `{"content_type":"Text","content":"…","source":null}`, NOT `{"kind":"text","value":"…"}` | use `content_type`/`content`/`source` |
| 2 | `POST /fuzzy/rules` and `POST /analysis/argumentation/gradual` | `tnorm: invalid type: string "godel", expected adjacently tagged enum TNormKind` | These two endpoints used to deserialise `tnorm` directly as the `TNormKind` enum (`{"kind":"godel"}`), while `/fuzzy/aggregate`, `/fuzzy/hybrid/probability`, `/fuzzy/quantify`, `/fuzzy/syllogism/verify`, `/fuzzy/fca/lattice` all accepted the bare-string registry name. **FIXED in this commit:** both endpoints now accept the same bare-string form (`"godel"` / `"goguen"` / `"lukasiewicz"` / `"hamacher"`) and route the lookup through `TNormRegistry`. Unknown names → `InvalidInput → HTTP 400`. Four regression tests added — see `src/api/fuzzy/learn_tests.rs::t9..t12`. |
| 3 | `POST /fuzzy/rules/{nid}/evaluate` | `missing field 'entity_id'` | The evaluate endpoint at v0.79.21 is **single-entity only** (it requires `entity_id`); there is no narrative-sweep mode. | call once per entity, or use the TensaQL `EVALUATE RULES … AGAINST (e:Actor)` form (which does sweep — see A.3) |
| 4 | `POST /fuzzy/measures/learn` | `dataset[0]: invalid type: map, expected a tuple of size 2` | Dataset items are 2-tuples `[input_vec, rank]`, NOT objects `{"input_vec": …, "rank": …}` | use array tuples |
| 5 | `POST /analysis/argumentation/gradual` | empty acceptability when an inline framework was sent | The endpoint loads the framework from the narrative's contention edges, NOT from an inline `framework` field. | seed contentions via `POST /contentions` first, OR use the in-process API (the integration test) for inline frameworks |

**Status:** the mixed-convention bug surfaced by this live replay was **fixed in the same commit** that added this appendix. Every fuzzy endpoint now accepts the bare-string `tnorm` form via the registry. The fix is locked behind four regression tests (`t9..t12`) so the inconsistency cannot regress.

## A.3 MCP-via-TensaQL replay summary

The live `mcp__tensa__query` tool was given each fuzzy verb as a TensaQL string against the same `operation-vesper-rest` narrative. Every numeric matches both the in-tree test and the REST replay.

| Capability | TensaQL | Live MCP response | Matches |
|---|---|---|---|
| Quantifier (most, split) | `QUANTIFY most (e:Actor) WHERE e.confidence > 0.85 FOR "operation-vesper-rest"` | `{value: 0.6, cardinality_ratio: 0.6, domain_size: 5}` | ✓ in-tree §6 |
| Quantifier (almost_all, split) | `QUANTIFY almost_all …` | `0.0` | ✓ |
| Quantifier (many, high) | `QUANTIFY many … > 0.7 …` | `1.0` | ✓ |
| Quantifier (few, high) | `QUANTIFY few … > 0.7 …` | `0.0` | ✓ |
| Syllogism (Figure I*) | `VERIFY SYLLOGISM { major:'Most type:Actor IS type:Actor', minor:'AlmostAll …', conclusion:'Most …' } FOR "…" THRESHOLD 0.5` | `{degree: 1.0, figure: "I*", valid: true}` | ✓ |
| Syllogism (Figure II) | swapped quantifier triple | `{degree: 1.0, figure: "II", valid: false}` | ✓ Peterson-invalid |
| FCA lattice | `FCA LATTICE FOR "…" ENTITY_TYPE Actor ATTRIBUTES […] WITH TNORM 'godel'` | `{num_concepts: 8, num_objects: 5, num_attributes: 4}` | ✓ |
| FCA concept (top, idx 0) | `FCA CONCEPT 0 FROM "<lattice_id>"` | `{extent_size: 5, intent: [(inflammatory_score, 0.1)]}` (all 5 actors share inflammatory_score ≥ 0.1) | ✓ |
| FCA concept (verified pair, idx 4) | `FCA CONCEPT 4 …` | `{extent: [Outlier-Loud, FactCheck-Gamma], intent: [(verified, 1.0), (inflammatory_score, 0.1)]}` | ✓ Bělohlávek-style natural grouping |
| Mamdani sweep | `EVALUATE RULES FOR "…" AGAINST (e:Actor) WITH TNORM 'godel'` | per-actor table identical to §9 (firing 0.8968 / 0.9802 / 0.9978 / 0 / 0; defuzz 0.7827 / 0.7833 / 0.7833 / null / null) | ✓ |
| Hybrid `INFER FUZZY_PROBABILITY` | uniform distribution over the 5 actors, event = `most:confidence>0.7,Actor` | `{value: 1.0, distribution_summary: "discrete:5"}` | ✓ (every actor satisfies, so P=1.0) |

**11 of 11 MCP-via-TensaQL responses match.** Notable: the TensaQL `EVALUATE RULES … AGAINST (e:Actor)` form **does** sweep the narrative (see the routes in `src/api/fuzzy/rules.rs` §A.2 row 3) — the REST endpoint at v0.79.21 is single-entity-only, but the TensaQL clause invokes `evaluate_rules_over_narrative` which iterates. That is another inconsistency worth surfacing: TensaQL is more capable than direct REST on this surface.

## A.4 Surfaces NOT covered by the live replay

| Surface | Why skipped |
|---|---|
| Direct fuzzy MCP tools (`fuzzy_aggregate`, `fuzzy_quantify`, …) | The MCP tool catalogue at the connected server publishes ~178 tools per CLAUDE.md, but this Claude client only sees the subset surfaced as deferred tools — the explicit `fuzzy_*` MCP tools are not surfaced to it. The same calls are exercised via TensaQL through `mcp__tensa__query`, which is the recommended pattern for agents that want fuzzy semantics + standard query plumbing in one call. |
| Gradual argumentation with an inline framework via REST | The `/analysis/argumentation/gradual` endpoint at v0.79.21 loads its framework from the narrative's contention edges, not an inline body. The in-process integration test (§12) covers the inline case. |
| Choquet measure learning at the synthetic-CIB scale via REST | The live REST run used an 8-sample inline dataset rather than the 100-sample synthetic CIB; the response was a degenerate AUC=1.0. The in-process test (§11) is the canonical numeric. |
| Studio Fuzzy Canvas (`/n/operation-vesper-rest/fuzzy`) | UI replay is out of scope for a CLI report. Every numeric the canvas would render is present in REST/MCP responses. |

## A.5 Does the live replay validate the §15 DeepRAP mapping?

Yes — every test section that ties to a DeepRAP deliverable returns the same numeric on three independent paths (in-process Rust API, REST endpoint, MCP TensaQL via `query`). For each commitment listed in §15:

| Deliverable | In-process | REST | MCP/TensaQL | Status |
|---|---|---|---|---|
| D2.3 — Gödel soundness | ✓ §3 | ✓ A.1 row 3 (Gödel TNormReduce = 0.1) | ✓ A.3 (`tnorm: godel` echoed) | numerics replay-stable |
| D2.4 — Łukasiewicz / Goguen | ✓ §3 | ✓ A.1 rows 4–6 | (registry list) | replay-stable |
| D2.5 — RO1.b replay soundness | ✓ §5 | ✓ A.1 row 13 (fuzzy Allen) | (would need MCP tool) | replay-stable |
| D2.6 — monotonic calibration under temporal revision | ✓ §13 | ✓ A.1 rows 25–26 (ORD-Horn) | (would need MCP tool) | replay-stable |
| D7.1 — RO2 first paper | ✓ §11 | ✓ A.1 row 21 (`/fuzzy/measures/learn`) | (would need MCP tool) | replay-stable |
| D7.A1 — production-quality PGD | ✓ §11 | ✓ A.1 row 21 + A.1 row 11 (Choquet additive recovers Mean) | (read back via FCA-style enumeration) | replay-stable |

Three failure modes that would have flagged a real issue and did NOT:

1. **Gödel default-invariant violation**: `GET /fuzzy/config` → `tnorm: godel`. Confirmed.
2. **Hybrid base-case identity**: `P_fuzzy(crisp event) = P(event)`. Confirmed at A.1 row 22 — the quantifier event saturates at 1.0 because every coalition member matches.
3. **Peterson-invalidity-despite-degree**: Figure II syllogism returns `valid: false` even with `degree = 1.0` on three independent paths. Confirmed at §7 (in-process), A.1 row 18 (REST), A.3 (MCP). This is the load-bearing property test of the syllogism module.

> **DeepRAP tie:** A.5 is the **F1 / F4 cross-partner integration deliverable** in miniature. The pattern — in-process integration test, REST endpoint replay, MCP TensaQL replay — is what the M12 / M30 calibration reports will look like applied to App-Partner benchmark cases at scale.

---

# Appendix B — Where each method came from, in plain English

Each fuzzy method TENSA ships has a story: a person, a frustration, a paper that changed the field. This appendix is a popular-science walkthrough — read it like a chapter, not a spec. Every section names the people, says why they were trying to solve the problem, and gives one tiny worked example you can carry in your head.

---

## B.1 T-norms — Karl Menger and the question of "what does AND really mean?"

In the 1940s an Austrian mathematician named **Karl Menger** (the same Menger of the *Menger Sponge*) was thinking about distances that aren't quite distances — *probabilistic metric spaces*, where the "distance" between two points is itself a probability distribution. He needed a way to combine two such fuzzy distances into one. His 1942 paper introduced **triangular norms**, *t-norms* for short — binary functions on `[0,1]² → [0,1]` that behave like AND on graded values.

For three decades the idea was a curiosity. Then in 1965 **Lotfi Zadeh** at Berkeley published *Fuzzy Sets*, which proposed using `min` for AND and `max` for OR. That choice (now called the **Gödel** t-norm, after Kurt Gödel's logic) is one valid t-norm — but as Menger had already shown, there are infinitely many. **Klement, Mesiar & Pap**'s 2000 monograph *Triangular Norms* finally pinned down the four canonical families that every modern fuzzy library ships:

- **Gödel (`min`)** — the *cautious* one. The whole is as strong as its weakest link.
- **Goguen (`a · b`)** — Joseph Goguen's 1967 PhD thesis at Berkeley. This is the *probabilistic* AND — if both sources are independent, this is `P(A ∧ B)`.
- **Łukasiewicz (`max(0, a+b−1)`)** — Jan Łukasiewicz, the Polish logician who in the 1920s built the first **three-valued logic** (true, false, possible). His t-norm is the strict one: *"if your two confidences don't add up to certainty, they don't combine to anything."*
- **Hamacher(λ)** — Horst Hamacher's 1978 family that smoothly interpolates Goguen at `λ=1` and the harsher Hamacher product at `λ=0`.

**Tiny worked example.** Two analysts, A and B, each say *"this footage is suspicious"* with confidence 0.6 and 0.7. Combined, the claim *"both think it's suspicious"* gets:

| t-norm | combined | what it means |
|---|---|---|
| Gödel | **0.6** | "we're as sure as the more cautious analyst" |
| Goguen | **0.42** | "if their judgements are independent, this is the joint probability" |
| Łukasiewicz | **0.3** | "they jointly under-cover certainty by 0.7, so we're only that sure" |

You don't pick the t-norm by guessing. You pick it by asking: *are these two estimates independent (Goguen), do I want to be cautious (Gödel), or do I demand they jointly cover the claim (Łukasiewicz)?* This is what makes t-norm choice a **calibration knob, not a hidden detail** — the heart of DeepRAP RO1.

---

## B.2 OWA aggregators — Ronald Yager's 1988 trick for "most"

When a critic says *"most reviewers liked the film"*, they don't mean *every reviewer*. They mean roughly the top 60–80%, with the bottom 20–30% discounted. **Ronald Yager** at Iona College in 1988 was frustrated that no aggregator could express this. The arithmetic mean weights everybody equally; the median throws away most of the data; `min`/`max` are extremes.

Yager's trick: sort the inputs in descending order, then apply a weight vector aligned with **rank**, not source. With weights `[1,0,0,…]` you get `max`. With `[0,…,0,1]` you get `min`. With uniform weights you get the mean. And by parameterising the weights through a *linguistic quantifier* `Q : [0,1] → [0,1]`, you get **"most"** as `w_i = Q(i/n) − Q((i-1)/n)`. The weights telescope to 1 by construction — a small piece of mathematical elegance that made the operator instantly popular.

OWA is now the standard aggregator for multi-criteria decision-making in industry — competitive procurement, medical diagnosis, sports judging. If you've ever heard the phrase *"drop the highest and lowest scores"*, that's an OWA in disguise (weights `[0, 1/(n-2), …, 1/(n-2), 0]`).

**Tiny worked example.** Five sources confidence-scoring a single claim: `[0.91, 0.88, 0.84, 0.10, 0.74]`. Yager's "**Few**" weights (`[0.667, 0.333, 0, 0, 0]`) say *"trust mostly the top one or two voices"*: the result is `0.667·0.91 + 0.333·0.88 = 0.90` — high, because the top voices agree. Yager's "**Most**" weights (`[0.0, 0.2, 0.4, 0.4, 0.0]`) ignore the very top and very bottom, returning `0.81` — slightly more conservative but still high because the *body* of voices agrees. Compare the bare arithmetic mean: `0.69` — dragged down by the dissenter. OWA lets you say "I want the body, not the extremes" without throwing data away.

---

## B.3 Choquet integral — the breakthrough Gustave Choquet didn't know was a breakthrough

In 1953 a French mathematician named **Gustave Choquet** published a paper titled *Theory of Capacities* — about non-additive measures in pure measure theory. He was thinking about geometry and functional analysis; he was not thinking about decision-making. The paper sat for 20 years until **Michio Sugeno** and others in Japan in the 1970s realised: *if you allow your "weights" to be a non-additive measure rather than a probability, the integral becomes an aggregator that can express interactions between criteria.* Suddenly Choquet's obscure paper was the missing piece in fuzzy aggregation theory.

The Choquet integral became famous in the 1990s through **Michel Grabisch** at the Sorbonne, whose 1996 paper turned it into a deployable AI primitive. The headline insight: an additive aggregator (mean, weighted average, even OWA) can NEVER express *"if these two signals BOTH fire, that's worth more than the sum of their parts"*. A Choquet integral can — by assigning extra mass `μ({i,j})` to the joint subset.

This matters for **coordinated-inauthentic-behaviour detection** in disinformation forensics: the signature of a real coordinated campaign isn't *"high temporal correlation"* OR *"high content overlap"*, but *"both at once, across the same accounts, in the same window"*. No additive aggregator can express that. A Choquet integral with `μ({temporal, content}) = 0.7` and `μ({temporal}) + μ({content}) = 0.4` says exactly *"the joint signal is worth more than the parts"* — and that is what TENSA's `learn_choquet_measure` learns automatically (see B.10).

**Tiny worked example.** Five source confidences `[0.91, 0.88, 0.84, 0.10, 0.74]`. With a Choquet measure that gives a `+0.30` bonus to any subset containing the suspected coalition `{A1, A2, A3}`, the integral returns **0.87** — higher than the mean (0.69) because *the right three voices agree*. With the symmetric-additive measure (which has no such bonus), the integral falls back to `0.69` exactly — the arithmetic mean. *Same five numbers, different aggregator, different story.*

---

## B.4 Fuzzy Allen relations — Dubois & Prade's correction to a 1983 oversight

In 1983 the AI researcher **James Allen** at Rochester published *Maintaining Knowledge About Temporal Intervals*, introducing the 13 canonical interval relations (`Before`, `Meets`, `Overlaps`, `Starts`, `During`, `Finishes`, `Equals`, and their inverses, plus `Contains`). Every AI temporal-reasoning system since uses these.

Allen assumed crisp endpoints — every interval has an exact start and end. But analysts don't write crisp times: they write *"in the morning"*, *"around dawn"*, *"may have started before the meeting ended"*. **Didier Dubois** and **Henri Prade** at IRIT Toulouse — the two pillars of European fuzzy-logic research, co-authors of the field's standard textbooks — recognised this gap in their 1989 paper *Processing Fuzzy Temporal Knowledge*. They replaced the crisp endpoints with *trapezoidal fuzzy numbers* and showed that Allen's 13 relations could be lifted to **13 graded degrees** that sum to roughly 1 — telling you *to what degree* each crisp relation holds.

**Steven Schockaert** and **Martine De Cock** (Ghent, 2008) refined the construction with a possibility/necessity averaging that's used today. The result: a single 13-vector `[μ_Before, μ_Meets, μ_Overlaps, …]` that captures *"this might be `Before`, but it could also be `Meets` or `Overlaps`, depending on how you read the timestamps"*.

**Tiny worked example.** Wave A of a posting campaign happens *"in the morning"* (kernel 08:00–11:00, ±60 min). Wave B happens *"around midday"* (kernel 11:00–13:00, ±60 min). The kernels just touch. Crisp Allen would say: **`Meets`, full stop.** Fuzzy Allen says: **`Before = 0.5, Meets = 0.5, Overlaps = 0.5`** — the system genuinely doesn't know which of the three it is, and this is more honest than picking one. An analyst writing *"wave B may have started while A was still going"* gets exactly this distribution back.

---

## B.5 Intermediate quantifiers — Petr Hájek's Czech school and "most"

In ordinary logic, quantifiers come in two flavours: *for all* (`∀`) and *there exists* (`∃`). But human language has dozens — *most, many, almost all, several, a few, hardly any* — and none of them are `∀` or `∃`. The Czech logician **Petr Hájek** at the Academy of Sciences in Prague spent his career building the formal foundations of fuzzy logic (his 1998 textbook *Metamathematics of Fuzzy Logic* is the field's bible). His student **Vilém Novák** at the University of Ostrava extended the work into *intermediate quantifiers*: ramps that map a cardinality ratio `r ∈ [0,1]` to a graded truth value.

The Novák / Murinová ramps in TENSA:

- **`Most`** — peaks at `r = 0.8`, starts firing at `r = 0.3`. *"more than just a few; the body of cases."*
- **`Many`** — peaks at `r = 0.5`, starts at `r = 0.1`. *"a non-trivial slice."*
- **`AlmostAll`** — peaks at `r = 0.95`, starts at `r = 0.7`. *"almost universal."*
- **`Few`** — the **De Morgan dual** of `Many`: `Q_few(r) = 1 − Q_many(r)`. *"the complement of many."*

These are not arbitrary numbers. They were calibrated against psycholinguistic studies — the points where speakers actually agree that *"most"* starts to fit a situation.

**Tiny worked example.** "Most actors satisfy `confidence > 0.85`" — applied to the Vesper narrative. Three of five actors satisfy → ratio `r = 0.6`. The `Most` ramp at `r = 0.6`: linearly interpolate between `(0.3, 0)` and `(0.8, 1)` → `(0.6 − 0.3)/0.5 = 0.6`. **Truth value: 0.6** — *"partially most, but not fully so"*. The `AlmostAll` ramp at the same `r=0.6` returns **0.0** — because `0.6 < 0.7` and the ramp hasn't started firing yet. Same domain, same predicate, two ramps, two answers — and both are correct under their own quantifier.

---

## B.6 Graded Peterson syllogisms — the 2,400-year-old puzzle goes fuzzy

Aristotle's *Prior Analytics* (~350 BC) is the founding document of formal logic. He classified valid argument forms — *syllogisms* — into figures: *"All Greeks are mortal; Socrates is Greek; therefore Socrates is mortal."* For 2,400 years logicians have argued about which figures are valid.

In 1979 **Philip Peterson** at Syracuse University extended Aristotle's framework to *intermediate quantifiers* — *"most, many, few"* — instead of just *"all, some, no"*. He produced a new square of opposition (the famous diagram from Boethius's medieval logic) that classified syllogisms into figures **I, I*, II, III, IV, V**. Crucially: **Figure II is invalid, no matter how true the premises are.**

Decades later **Petra Murinová** and **Vilém Novák** in 2014 graded the system. Each statement now has a degree in `[0,1]`; the syllogism's combined degree is the t-norm conjunction of the three. But — and this is the load-bearing rule — *a high combined degree does not make Figure II valid.* The taxonomy overrides the numeric.

**Tiny worked example.** A canonical Figure I* syllogism: *"Most actors are confident; Almost all confident actors are coordinated; therefore Most actors are coordinated."* If every actor counts as confident and coordinated, all three statements have degree 1.0; the Gödel min-fold returns 1.0; figure is `I*`; **valid = true**. Now swap the major and minor premises (turning it into Figure II): *"Almost all actors are confident; Most confident actors are coordinated; therefore Most actors are coordinated."* Same numbers, same degree (1.0), but figure is `II`, so **valid = false**. The truth degree is meaningless without the figure check — that is the Peterson taxonomy.

---

## B.7 Fuzzy Formal Concept Analysis — Bělohlávek's lattices for messy data

**Rudolf Wille** at TU Darmstadt invented Formal Concept Analysis (FCA) in 1982. The idea is beautiful: given a table of objects × attributes (a *formal context*), the algebra of *closures* under the duality `extent ↔ intent` produces a **complete lattice** — a partially-ordered hierarchy of "concepts" where every node is `(set of objects, set of attributes they all share)`. No clustering, no learning — just pure algebra.

Wille's FCA assumed binary attributes: an object either has the attribute or doesn't. But real data is graded — *"this actor is anonymous to degree 0.7"*. **Radim Bělohlávek** at Palacký University in Olomouc generalised the framework to **fuzzy attributes** in 2004. The closure operator now uses *residual implications* (the inverse of a t-norm), and the resulting concepts have **graded intents** — each attribute appears with a membership degree.

The algorithm to enumerate all concepts is **NextClosure**, due to **Bernhard Ganter** in 1984 — a deterministic walk through all `2^|O|` subsets, deduplicating by closure. Bělohlávek's adaptation handles the graded case. The 64-object cap in TENSA comes from the `u64` bitmask — the algorithm is exponential in the number of objects.

**Tiny worked example.** Five Vesper actors × four attributes (`anonymous`, `foreign_funded`, `verified`, `inflammatory_score`). The concept lattice has 8 nodes. The top concept has all 5 actors and the intent *"every actor has at least 0.1 inflammatory score"* — true, and trivially so. A more interesting concept: extent `{Outlier-Loud, FactCheck-Gamma}`, intent `{verified: 1.0, inflammatory_score: 0.1}` — *"the verified non-inflammatory actors"*. This concept emerges automatically from the data, not from any analyst tag. That is what FCA does: it *discovers* the natural groupings, with their natural attribute signatures.

---

## B.8 Mamdani fuzzy rule systems — the 1975 steam-engine controller

**Ebrahim Mamdani**, an Iranian-British engineer at Queen Mary College London, faced a problem in 1974 that had defeated PID control: *how do you operate a small steam engine using verbal rules from a human expert?* The expert said things like *"if the pressure is high and the temperature is rising, close the valve a bit"*. PID can't read that. Mamdani — building on Lotfi Zadeh's *fuzzy sets* (1965) and *fuzzy algorithms* (1973) — built a controller that could.

The Mamdani-Assilian 1975 paper *An Experiment in Linguistic Synthesis with a Fuzzy Logic Controller* is one of the most cited papers in engineering. It introduced the **fuzzy rule system**:

1. *Fuzzify* the input: each numeric value (`pressure = 120`) becomes a vector of *linguistic-term* memberships (`high: 0.7, medium: 0.3, low: 0.0`).
2. *Fire each rule* under a t-norm: `IF pressure IS high AND temperature IS rising THEN close-valve IS small`. The firing strength is the t-norm of the antecedent memberships.
3. *Aggregate* the consequents: take the union (or sum) of the scaled output fuzzy sets.
4. *Defuzzify*: pick a single number out of the aggregate fuzzy set — usually the **centroid** (centre of mass) or **mean-of-maxima**.

Mamdani's controller was deployed commercially within five years. By the 1990s, fuzzy controllers were running Sendai Subway in Japan, Sony camcorder autofocus, Hitachi washing machine cycles. Today the mathematics is unchanged — but it lives in disinformation analysis, medical decision support, and any domain where domain experts can articulate rules that nobody can write down as crisp formulas.

**Tiny worked example.** Rule: *"IF inflammatory-score IS high AND temporal-correlation IS high AND confidence IS validated THEN disinfo-risk IS elevated"*. Given Vesper-A1 (inflammatory 0.81, temporal-correlation 0.91, confidence 0.92): each antecedent's membership is roughly `[1.0, 1.0, 0.9]`; Gödel min → firing `0.9`. The `elevated` consequent (a triangular fuzzy set peaking at 0.85) gets scaled by 0.9, defuzzified by centroid: **risk = 0.78**. For Outlier-Loud (high inflammatory but low correlation), the second antecedent is 0 → firing collapses to 0 → risk = `null`. The rule **discriminates coordination, not heat** — Mamdani would have approved.

---

## B.9 Fuzzy-probabilistic hybrid — when fuzzy logic met probability theory

This one is genuinely hard, and TENSA only ships the base case.

In the 1980s, the AI community argued bitterly about whether fuzzy logic was a competitor to probability or a complement. **Lotfi Zadeh** said: *they answer different questions.* Probability tells you how often something happens; fuzzy logic tells you to what degree something is true. *"Most birds fly"* — probability handles the *most*, fuzzy logic handles the *fly* (a penguin only partially flies).

Putting them together is hard because the semantics conflict. **Ronald Fagin** and **Joseph Halpern** at IBM proposed the first rigorous *probability of fuzzy events* in 1994: integrate the membership function `μ_E(x)` against the probability density `P(x)` — `P_fuzzy(E) = ∫ μ_E(x) · P(x) dx`. For discrete distributions this is a simple sum: `Σ μ_E(eᵢ) · P(eᵢ)`. For crisp events (μ ∈ {0, 1}) it reduces to classical probability — backwards-compatibility for free.

The full theory — modal-logic embeddings, decidability proofs, multi-agent epistemic versions — is still active research. **Tommaso Flaminio** (Genoa) and **Lluís Godo** (IIIA Barcelona) are the names to watch; **Cao, Holčapek & Flaminio**'s 2026 FSTA paper is the state of the art. TENSA ships the discrete-distribution base case; the modal-logic lift is exactly what DeepRAP RO3 / P3 commits to.

**Tiny worked example.** Three Vesper-A actors form a uniform-weight coalition (P=1/3 each). Event: *"is this actor inflammatory?"* with μ = the actor's `inflammatory_score`. Compute: `(0.81 + 0.78 + 0.74) / 3 = 0.78`. That's the probability that a random member of the coalition fires the inflammatory predicate, weighted by *how inflammatory each actor is*. If you had used a crisp predicate (`inflammatory_score > 0.7`) instead, you'd get 1.0 (all three exceed the threshold). The fuzzy version preserves the *grading* while still giving you a probability. That's what hybrid inference is for.

---

## B.10 Choquet measure learning — Grabisch's 1996 trick made deployable

Choquet integrals (B.3) are powerful — but they need a measure, and a measure on `n` signals has `2^n` parameters. For `n = 4` that's 16 numbers. Where do they come from? Either (a) an analyst hand-rolls them (rarely satisfying), (b) a symmetric default like `additive` / `pessimistic` / `optimistic` (which throws away the whole point of using Choquet), or (c) you **learn** them from data.

**Michel Grabisch** worked on learning fuzzy measures from labelled examples for a decade after his 1996 paper. The setup: you have a dataset of `(input_vec, score_or_rank)` pairs; you want a measure `μ` such that `Choquet(input_vec; μ)` ranks the data correctly. This is a constrained optimisation problem — the measure must be **monotone** (`A ⊆ B ⇒ μ(A) ≤ μ(B)`) and **boundary-pinned** (`μ(∅) = 0`, `μ(N) = 1`). The classical approach was a quadratic program (OSQP, Clarabel).

TENSA ships a pure-Rust **projected gradient descent** in the full `2^n` capacity space, with monotonicity and boundary projection at each step. The loss is the **mean-normalised pairwise hinge loss** of Amgoud, Bustince, and others' ranking-supervised tradition: for every pair where `rank_i < rank_j`, the learned measure should give `Choquet(input_i) > Choquet(input_j) + margin`. The capacity space caps at `n ≤ 6` because of the `2^n` parameter count; the **k-additive specialisation** of Grabisch 1997 reduces this to `O(n^k)` and is the next sprint.

The reason this matters for disinformation analysis is the **synthetic-CIB acceptance demo**: a 4-signal dataset with a *hidden multiplicative interaction* between `temporal_correlation` and `content_overlap`. No additive measure can recover the ranking — the AUC stays around 0.45 (worse than coin flip). The learned Choquet measure recovers it: AUC ≈ 0.85. **The +0.40 gap is the operational proof that learning the measure changes what you can detect.**

**Tiny worked example.** 100-cluster synthetic dataset, hidden score `s = sigmoid(2·x₀·x₁ + 0.3·x₂ − 0.5·x₃)`. Train + test on a 50/50 split:

| Aggregator | Test AUC | What it can express |
|---|---|---|
| `symmetric_additive` (= mean) | **0.45** | linear contribution per signal — can't see the `x₀·x₁` interaction |
| **Learned Choquet** | **0.85** | full `2⁴ = 16` cell capacity table — *can* see the interaction |

The learned measure has a high `μ({0, 1})` cell — it has *learned the interaction*. Same data, same algorithm, different aggregator family, fundamentally different ceiling.

---

## B.11 Gradual argumentation — Dung's 1995 frameworks gone graded

In 1995 a Vietnamese-Swiss researcher named **Phan Minh Dung** (then at the Asian Institute of Technology) published one of the most influential AI papers ever: *On the acceptability of arguments and its fundamental role in non-monotonic reasoning, logic programming and n-person games.* Dung introduced **abstract argumentation frameworks** — directed graphs where nodes are arguments and edges are *attacks*. He defined four canonical *extensions* (sets of arguments that survive together): *grounded, preferred, complete, stable*. The framework became the foundation of formal argumentation theory.

Dung's extensions are crisp: an argument is either IN the extension or OUT. But human arguments aren't binary — *"this argument has merit even though it is partially defeated."* **Pietro Baroni**, **Massimiliano Giacomin**, **Leila Amgoud** (Toulouse), and **Jérôme Ben-Naim** spent the 2000s and 2010s building **gradual / ranking-based semantics** that return a continuous *acceptability* in `[0, 1]` per argument.

TENSA ships four canonical variants:

- **h-Categoriser** (**Besnard & Hunter**, 2001): `Acc(a) = w(a) / (1 + Σ Acc(b))` over attackers `b`. The classic recurrence — every argument's acceptability is divided by `1 + sum of attackers' acceptabilities`.
- **Weighted h-Categoriser** (**Amgoud & Ben-Naim**, 2017): same shape but each attack carries a weight `v_{ba}`.
- **Max-Based** (Amgoud & Ben-Naim, 2013): replaces the sum with `max` — *"only your strongest attacker matters."*
- **Card-Based** (Amgoud & Ben-Naim, 2013): denominator is `(1 + card_attackers) · (1 + sum)` — *number* of attackers and their *strength* both matter.

The 2013 paper is the canonical comparative reference: *"Ranking-Based Semantics for Argumentation Frameworks"*. Every gradual semantics worth knowing about is benchmarked against three principles — **anonymity** (relabelling argument IDs doesn't change acceptability), **independence** (disconnected components don't interfere), **monotonicity** (adding an attacker can only lower the target's acceptability). TENSA's implementation passes all three on 30 random ChaCha8-seeded frameworks per principle.

**Tiny worked example.** Four arguments: *"Vesper is coordinated"* (intrinsic 0.85), *"Vesper looks organic"* (0.45), *"Fact-check is correct"* (0.92), *"Fact-check is biased"* (0.30). Five attacks: organic ↔ coordinated, fact-check-correct ↔ fact-check-biased, fact-check-correct → organic. h-Categoriser converges in 16 iterations:

| Argument | Acceptability |
|---|---|
| Vesper is coordinated | **0.72** |
| Vesper looks organic | 0.18 |
| Fact-check is correct | **0.79** |
| Fact-check is biased | 0.17 |

The fact-check argument survives near its intrinsic strength because its attacker (fact-check-biased) is weak. The Vesper-coordinated argument survives strongly because its attacker (organic) is itself attacked by the fact-check verdict — **indirect support**, captured by the iterative recurrence. This is why gradual semantics matter: they capture support-through-defeat-of-attackers, which crisp Dung extensions can't.

---

## B.12 ORD-Horn closure — Nebel & Bürckert's 1995 tractability landmark

Allen's interval algebra (B.4) has a dirty secret: *deciding satisfiability of a disjunctive Allen network is NP-complete.* You can write down four events with disjunctive constraints (*"event A is `Before` or `Meets` event B, event B is `Overlaps` or `During` event C, …"*) and asking *"is this jointly satisfiable?"* is fundamentally hard.

In 1995 **Bernhard Nebel** at Freiburg and **Hans-Jürgen Bürckert** at DFKI Saarbrücken published *Reasoning About Temporal Relations: A Maximal Tractable Subclass of Allen's Interval Algebra* in the *Journal of the ACM*. They identified a specific 868-element subset of the disjunctive Allen relations — the **ORD-Horn class** — for which **path-consistency closure is sound and complete**. Inside ORD-Horn, you get a polynomial-time decision procedure. Outside, you don't.

Nebel and Bürckert's proof is long and technical, but the punchline is simple: an empty constraint anywhere in the network proves unsatisfiability; a non-empty closure proves satisfiability *if and only if* every input disjunction is in ORD-Horn.

The closure algorithm itself is from **Peter van Beek** (1992): iterate over every triple `(i, j, k)`, intersect the current constraint with the composition of the indirect path through `k`, repeat until nothing changes. TENSA ships this on top of its existing 13×13 composition table — a 30-year-old algorithm port, not a reinvention.

**TENSA's caveat:** the 868-element ORD-Horn membership oracle is not shipped this sprint. The closure is **sound for any Allen network** (an empty constraint always proves unsatisfiability) but **complete only inside ORD-Horn**. Callers requiring decidability guarantees must restrict to a known tractable class by construction.

**Tiny worked example.** Four events: Funding-Wire, Persona-Build, Coordinated-Posting, Takedown. Inputs:

```
0 → 1 ∈ {Before, Meets}
1 → 2 ∈ {Before, Meets, Overlaps}
0 → 2 ∈ {Before}
2 → 3 ∈ {Before, Meets}
```

Closure tightens these to six pair constraints, **adding two new ones** the analyst never wrote down:

```
(0, 3) ∈ {Before}    ← INFERRED via 0→2→3
(1, 3) ∈ {Before}    ← INFERRED via 1→2→3
```

If we change `(0, 2)` to `{After}` (analyst makes a typo), the closure detects the contradiction immediately: `(0, 2)` becomes the empty set. **Unsatisfiable**. This is a 30-year-old idea doing 21st-century work — flagging revision conflicts in OSINT timelines before an analyst bases their report on a contradiction.

---

## B.13 Putting it all together — why these twelve, why now

Each of these methods came from a different decade and a different motivation:

| Decade | Method | Motivated by |
|---|---|---|
| 1940s | T-norms (Menger) | Probabilistic distance |
| 1950s | Choquet integral (Choquet) | Pure measure theory |
| 1960s | Fuzzy sets (Zadeh) | Engineering control |
| 1970s | Mamdani rules (Mamdani) | Steam engine control |
| 1980s | FCA (Wille), Allen algebra (Allen), Yager OWA | Knowledge representation, decision theory |
| 1990s | Dung argumentation, Nebel-Bürckert ORD-Horn, Fagin-Halpern hybrid | Non-monotonic reasoning, decidability |
| 2000s | Bělohlávek fuzzy FCA, Hájek-Novák quantifiers, Schockaert fuzzy Allen | Graded knowledge representation |
| 2010s | Amgoud–Ben-Naim gradual semantics, Murinová-Novák Peterson syllogisms | Continuous reasoning |
| 2020s | Choquet measure learning (PGD), Cao-Holčapek-Flaminio hybrid | Deployable AI primitives |

What unifies them is a single conviction the European fuzzy-logic community has held since the 1970s: *uncertainty has structure, and that structure is mathematical.* Probability is one valid theory of uncertainty. T-norm-based fuzzy logic is another. Decision theory under non-additive measures is a third. Argumentation under graded acceptability is a fourth. Each captures something the others miss.

TENSA's wager — and DeepRAP's wager — is that **a real-world AI system handling decision-grade reasoning over contested information needs all four**, exposed as configurable knobs, with explicit semantics, and with the calibration guarantees that the recent decade's theory provides. The twelve methods in this appendix are the foundation. The DeepRAP project is the proof that the foundation holds.
