---
name: tensa-synth
description: Synthetic hypergraph generation via EATH and other surrogate models — activate when the user asks about null models, scaling benchmarks, significance tests, wargame substrates, or anything under /synth/*.
---

# TENSA Synthetic Generation Assistant

Activate this skill when the user wants to:

- **Test whether an observed structure is significant** ("is this
  community real or just chance?", "is this motif over-represented?")
- **Build a null model** for pattern mining, community detection,
  temporal motifs, or higher-order contagion.
- **Stress-test analysis algorithms** at scale without real corpora.
- **Generate adversarial wargame substrates** that look like calibrated
  civilian backgrounds.
- **Explore the EATH parameter space** (`rho_low`, `rho_high`, `xi`,
  `p_from_scratch`, `omega_decay`) to understand burstiness or
  group-recruitment dynamics.
- **Reproduce a synthetic run** from its `ReproducibilityBlob` (seed
  + params + chunk hashes).

The first concrete model is **EATH** (Effective Active Temporal
Hypergraph) from Mancastroppa, Cencetti, Barrat — arXiv:2507.01124v2.
Higher-order contagion uses Iacopini, Petri, Barrat, Latora,
Nat Commun 2019.

## When to suggest synth

| User signal | Why synth |
|---|---|
| "Is the community I see real?" | Run `INFER COMMUNITIES` on K=100 surrogates → compare modularity z-score. |z| ≥ 1.96 ⇒ p ≤ 0.05. |
| "Is this temporal motif over-represented?" | `metric: "temporal_motifs"`, K=100 — per-motif z-score. |
| "I need 5000 entities for a benchmark." | Calibrate EATH on a real narrative, then generate a scaled-up version. Keeps the burstiness + group-size profile honest. |
| "Run a wargame on a fake civilian background." | `BackgroundSubstrate::Synthetic` (or `SyntheticHybrid` to mix corpora). |
| "Does this pattern beat random?" | `metric: "patterns"` — PRESENCE/ABSENCE z-scores per pattern. |
| "What if this disinformation cascade hit a different network?" | `INFER HIGHER_ORDER_CONTAGION(...)` on the existing narrative, OR `/synth/contagion-significance` for the surrogate-vs-real contrast. |
| "I have a corpus and want to find hidden coordination groups I didn't ingest." | INVERSE problem — use Phase 15c **hypergraph reconstruction** (`INFER HYPERGRAPH FROM DYNAMICS FOR "..."`), NOT a surrogate model. See the dedicated `tensa-reconstruction` skill bundle. Reconstruction recovers latent hyperedges from observed entity time-series; surrogates generate synthetic hyperedges from a calibrated null model. Don't confuse the two. |

If the user asks for "synthetic data for training an LLM" — DON'T
suggest synth. EATH preserves graph statistics, not semantic content;
participations are `Role::Bystander`, info_sets are empty. It's a
null-model generator, not a data-augmentation tool.

## Interpreting z-scores

The significance pipeline computes:

```
z = (observed_value - mean(K_surrogates)) / stddev(K_surrogates)
p = two-sided normal p-value
```

| |z| | p (≈) | Interpretation |
|---|---|---|
| < 1.0 | > 0.32 | Indistinguishable from null. The observed structure could plausibly arise by chance under EATH. |
| 1.0 — 1.96 | 0.05 — 0.32 | Suggestive but not significant. Mention to user; don't claim discovery. |
| ≥ 1.96 | ≤ 0.05 | **Significant at the standard threshold.** The observed structure is genuinely above the EATH null. |
| ≥ 2.58 | ≤ 0.01 | Strongly significant. |
| ≥ 3.29 | ≤ 0.001 | Extremely significant — but also worth checking K is large enough (K ≥ 100). |

**Caveats to surface to the user:**

- A single null model isn't enough for confident discovery. Industry
  practice is to run BOTH EATH (preserves activation rhythm) and a
  configuration-style null (preserves degree). Phase 13 (`"nudhy"`
  surrogate) lands the second model.
- `K=100` is the default; clamp at 1000. K below 30 produces unstable
  z-scores — warn the user.
- Default thresholds in `FidelityThresholds` are PLACEHOLDER values
  (see Phase 12.5 follow-up). Reports rendered with default thresholds
  carry a `⚠ Default` warning banner.

## EATH parameter intuition

EATH treats actors as activity-modulated stations. A short cheat-sheet:

| Param | Effect | Calibration source |
|---|---|---|
| `rho_low` | Probability per step of leaving the **quiet** state. Lower = longer quiet bursts (heavier-tailed inter-event distribution). | Estimated from per-entity quiet-run lengths in source narrative. Clamped `[0.01, 1.0]`. |
| `rho_high` | Probability per step of leaving the **active** state. Lower = longer active bursts. | Same, from active runs. |
| `xi` | Mean number of groups per Λ_t time bucket. Drives total event volume. | Mean groups per bucketed activity multiplier. Clamped `[0.1, 50.0]`. |
| `p_from_scratch` | Per-recruitment probability of building a new group from scratch (vs. mutating from short-term memory). High = low overlap; low = sticky cliques. | `1 − (consecutive-pair ≥50% overlap fraction)`. |
| `omega_decay` | Long-term-memory exponential-decay constant. Higher = faster forgetting. | Currently default; per-entity fitting deferred. |
| `xi` (high) + `p_from_scratch` (low) | "Inner circle" dynamics. | — |
| `xi` (low) + `p_from_scratch` (high) | "Crowd" dynamics. | — |
| `stm_capacity` | Short-term memory ring buffer size (per entity). | Default 7 (Miller's number, deliberate). Increase for longer working-memory analogues. |
| `max_group_size` | Hard cap on hyperedge size at recruitment. | Max observed in source, capped 50. |
| `aT[i]` | Per-entity activity rate. Long vector. | Per-entity participation rate in source. |
| `Λ_t` | Time-bucketed activity multiplier (≤100 buckets). | Time-bucket totals in source, normalized. |

If the user's calibration produces a **fidelity report** with metrics
failing thresholds — usually `inter_event_ks` or `burstiness_mae` — the
fix is rarely a parameter override. The EATH model is a generative
*approximation*, not a faithful replicator. Tell them: "the fit is
within EATH's expected envelope; for replication-grade fidelity wait
for Phase 13 NuDHy or the threshold-calibration follow-up."

## /synth/* REST endpoints quick reference

| Method | Path | Use |
|---|---|---|
| `POST` | `/synth/calibrate/{narrative_id}` | Submit calibration job. Body: `{model: "eath"}` (default). Returns `{job_id}`. |
| `GET` | `/synth/params/{nid}/{model}` | Read calibrated params for a narrative+model pair. |
| `PUT` | `/synth/params/{nid}/{model}` | Override params manually (advanced). |
| `DELETE` | `/synth/params/{nid}/{model}` | Drop the calibrated params. |
| `POST` | `/synth/generate` | Submit generation job. Body: `{source_narrative_id, output_narrative_id, model?, params?, seed?, num_steps?, label_prefix?}`. |
| `POST` | `/synth/generate-hybrid` | Submit hybrid (mixture) generation. Body: `{components:[{narrative_id, model, weight}], output_narrative_id, seed?, num_steps?}`. Weights must sum to 1.0 ± 1e-6. |
| `GET` | `/synth/runs/{nid}` | List runs for a narrative (`?limit=N`, default 50, newest first). |
| `GET` | `/synth/runs/{nid}/{run_id}` | Single run summary. |
| `GET` | `/synth/seed/{run_id}` | `ReproducibilityBlob` (seed + params + chunk hashes). |
| `GET` | `/synth/fidelity/{nid}/{run_id}` | `FidelityReport` for a run. Fidelity reports accept an OWA aggregator via `?aggregator=owa&weights=...` to weight the harder KS tests more heavily (Fuzzy Sprint, v0.78.0). |
| `GET` | `/synth/fidelity-thresholds/{nid}` | Per-narrative threshold config. |
| `PUT` | `/synth/fidelity-thresholds/{nid}` | Override thresholds for a narrative. |
| `GET` | `/synth/models` | List registered surrogate models (`["eath", ...]`). |
| `POST` | `/synth/significance` | Submit significance job. Body: `{narrative_id, metric: "temporal_motifs"\|"communities"\|"patterns", k?, params_override?}`. |
| `GET` | `/synth/significance/{nid}/{metric}/{run_id}` | Single significance result. |
| `GET` | `/synth/significance/{nid}/{metric}` | List significance results for a metric. |
| `POST` | `/synth/contagion-significance` | Higher-order SIR significance. Body: `{narrative_id, params: HigherOrderSirParams, k?, model?}`. |
| `GET` | `/synth/contagion-significance/{nid}/{run_id}` | Single contagion-significance result. |
| `GET` | `/synth/contagion-significance/{nid}` | List contagion-significance results. |
| `POST` | `/analysis/higher-order-contagion` | **Synchronous** higher-order SIR on a real narrative. No surrogates, no K-loop, just the simulation. |

## MCP tools (7)

| Tool | Wraps | Notes |
|---|---|---|
| `calibrate_surrogate` | `POST /synth/calibrate/{nid}` | Args: `{narrative_id, model? = "eath"}`. Returns `{job_id, status}`. |
| `generate_synthetic_narrative` | `POST /synth/generate` | All Phase-6 fields supported. |
| `generate_hybrid_narrative` | `POST /synth/generate-hybrid` | Validates Σ weight = 1.0 ± 1e-6 synchronously. |
| `list_synthetic_runs` | `GET /synth/runs/{nid}` | `?limit=` clamped to [1, 1000]. |
| `get_fidelity_report` | `GET /synth/fidelity/{nid}/{run_id}` | 404 → null. |
| `compute_pattern_significance` | `POST /synth/significance` | `metric` ∈ `temporal_motifs`/`communities`/`patterns`. Routes higher-order contagion to the dedicated tool. |
| `simulate_higher_order_contagion` | `POST /synth/contagion-significance` | Pass `HigherOrderSirParams` opaquely; engine deserializes. |

## Provenance: synthetic vs empirical

Every synthetic record carries:

- `properties.synthetic = true`
- `properties.synth_run_id = <uuid>`
- `properties.synth_model = "eath"` (or other model name)
- `entity.extraction_method = "Synthetic"` (or `Situation.extraction_method`)
- `time_granularity = "Synthetic"` on situations
- Participation provenance via `info_set.knows_before` sentinel
  `KnowledgeFact{about_entity: NIL_UUID, fact: "synthetic|run_id=…|model=…"}`

**Default behavior:** read endpoints filter out synthetic records.
Opt-in via:

- TensaQL: `INCLUDE SYNTHETIC` clause (where supported)
- REST: `?include_synthetic=true` query param
- Studio: ⊛ button in workspace header (purple = on)

Endpoints that respect the flag today (5):
`/narratives/:id/stats`, `/entities`, `/situations`, `/ask` (body
field accepted; downstream RAG context filtering pending Phase 12.5),
`/export/archive`.

## Phase 12.5 deferred items — DO NOT promise these

The following are KNOWN GAPS (logged for the next sprint). Don't tell
the user any of these are working today:

1. **Reproduce drift detection.** The Studio Reproduce dialog shows a
   static "may not match exactly" disclaimer. Live drift detection via
   `GET /synth/state-hash/{nid}` is not yet wired (endpoint doesn't exist).
2. **`/ask` RAG context filtering.** The route accepts
   `include_synthetic: bool` but the downstream RAG context assembly
   (`crate::query::rag`) currently discards it. Default behaviour is
   correct (synthetic excluded by default), but explicit opt-in won't
   change retrieval today.
3. **`/narratives/:id/communities` known leak.** Community summaries
   are pre-rendered at write time without distinguishing
   synthetic-derived from empirical. Fix: add `synthetic_derived: bool`
   to `CommunitySummary` (Phase 12.5).
4. **7 PENDING aggregation endpoints** at the spec'd EATH path don't
   exist in TENSA today — when they land, they MUST add the
   `?include_synthetic=true` opt-in:
   - `POST /analysis/centrality` (TENSA exposes per-entity virtual
     props + `POST /jobs`)
   - `POST /analysis/communities` (TENSA exposes
     `POST /narratives/:id/communities/summarize`)
   - `POST /analysis/temporal-motifs` (only `GET /narratives/:id/temporal-motifs` readback exists)
   - `POST /analysis/contagion` (same readback shape)
   - `GET /fingerprint/stylometry/{nid}` (TENSA: `GET /narratives/:id/fingerprint`)
   - `GET /fingerprint/disinfo/{nid}` (disinfo-feature gated)
   - `GET /fingerprint/behavioral/{nid}` (TENSA: per-entity, not per-narrative)
5. **Cancellation tokens for synth jobs.** The `InferenceEngine` trait
   has no cancellation token; `DELETE /jobs/:id` cancels Pending only,
   not Running. T6 of `synth::engines_tests` is `#[ignore]`'d.
6. **HAD / NuDHy / hyperedge-config surrogates.** Phases 13–15 of the
   research extension. Currently only `"eath"` is registered.
7. **Per-model calibration dispatch.** `SurrogateCalibrationEngine`
   rejects any `model.name() != "eath"` because
   `calibrate_with_fidelity_report` is EATH-specific. Future surrogate
   families need a generalized fidelity surface.
8. **Threshold calibration study.** The default `FidelityThresholds`
   are vibes-based PLACEHOLDER values. A study fitting them against
   the Mancastroppa paper's RFID datasets + 3-5 real TENSA narratives
   is logged as a follow-up.

If the user asks for any of the above, tell them it's on the Phase
12.5 / extension roadmap and offer the closest currently-shipping
substitute.

## Don't

- **Don't run significance with `K < 30`.** The z-score will be unstable.
- **Don't claim "p < 0.05" without K ≥ 100.** It's noise at lower K.
- **Don't suggest synth for prose, RAG answers, or anything semantic.**
  Redirect to real narratives.
- **Don't override `params` without understanding the effect.** The
  fitted params from `CALIBRATE SURROGATE` are usually the right
  starting point; manual overrides are an advanced tuning knob.
- **Don't promise reproducibility without saving the seed.** Always
  pass `SEED <n>` for reproducible runs and surface the
  `ReproducibilityBlob` URL (`GET /synth/seed/{run_id}`) to the user.
- **Don't generate hybrids whose weights don't sum to 1.0.** The
  validator rejects with a 400; tell the user the constraint.
- **Don't conflate `/analysis/higher-order-contagion` (synchronous)
  with `/synth/contagion-significance` (async, K surrogates).** First
  is "run SIR once on this network"; second is "is the observed SIR
  output significant against the EATH null?".
