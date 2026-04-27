---
name: tensa-reconstruction
description: SINDy-based hypergraph reconstruction — activate when the user wants to recover latent coordination structure from observed entity time-series (the inverse problem to synth generation).
---

# TENSA Hypergraph Reconstruction Assistant

Activate this skill when the user wants to:

- **Discover hidden coordination groups** from a disinfo / Telegram /
  social-media corpus where group structure was never explicitly
  ingested.
- **Validate ingested coordination claims** against what the data
  actually supports (cross-reference declared groups with inferred
  edges).
- **Quantify how much group structure is observable** in a particular
  observation source (participation rate, sentiment, engagement) by
  measuring AUROC against EATH-generated ground truth.
- **Materialize inferred hyperedges as Situations** for downstream
  analysis (centrality, contagion, wargaming) under their own
  reconstruction provenance.

This is the **INVERSE** of synthetic generation — important enough to
have its own skill bundle. Synth generates synthetic hyperedges from a
calibrated null model. Reconstruction recovers latent hyperedges from
observed entity dynamics. Different problem, different math, different
provenance.

## Method

THIS / SINDy method of Delabays, De Pasquale, Dörfler, Zhang —
*Hypergraph reconstruction from dynamics*, Nat. Commun. 16:2691 (2025),
arXiv:2402.00078. Uses Brunton et al.'s SINDy (PNAS 113:3932, 2016) as
the parent framework with a Pearson-pre-filtered monomial library +
LASSO + symmetrization + bootstrap for confidence.

Pairwise baseline: Casadiego, Nitzan, Hallerberg, Timme — ARNI, Nat.
Commun. 8:2192 (2017). Reconstruction generalizes ARNI to higher-order
group interactions via Taylor expansion of the dynamics around a
reference point.

## TensaQL surface

```sql
-- Defaults: observation = participation_rate, max_order = 3, λ auto.
INFER HYPERGRAPH FROM DYNAMICS FOR "telegram-corpus-1"

-- Tunable form.
INFER HYPERGRAPH FROM DYNAMICS FOR "tg-corpus"
    USING OBSERVATION 'participation_rate'   -- only fully-implemented source
    MAX_ORDER 3                              -- 2..=4
    LAMBDA 0.05                              -- override λ_max heuristic
```

Returns `{ job_id, status: "submitted" }`. Poll via the standard `/jobs`
or the dedicated `/inference/hypergraph-reconstruction/{job_id}` GET.

## REST surface

```
POST /inference/hypergraph-reconstruction
  body: { narrative_id, params?: ReconstructionParams }
  → 201 { job_id, status: "Pending" }

GET  /inference/hypergraph-reconstruction/{job_id}
  → 200 InferenceResult { result: { kind: "reconstruction_done",
                                    result: ReconstructionResult } }

POST /inference/hypergraph-reconstruction/{job_id}/materialize
  body: { output_narrative_id, opt_in: true, confidence_threshold?: f32 }
  → 200 MaterializationReport { situations_created, ... }
```

Materialization writes one Situation per inferred hyperedge with
`confidence > threshold` (default 0.7). Each situation carries
`extraction_method = ExtractionMethod::Reconstructed { source_narrative_id, job_id }`.
Per-situation refs live under `syn/recon/{output_narrative_id}/{job_id}/{situation_id}`.

## MCP

```
reconstruct_hypergraph(narrative_id, params?) → { job_id, status }
```

Identical envelope to the REST `submit` endpoint.

## Analyst workflow — confidence > 0.7, NOT weight > ε

Per architect §13.7 of `docs/synth_reconstruction_algorithm.md`: The
Taylor expansion makes triadic terms contribute nonzero pairwise
coefficients. A pairwise edge with weight above the LASSO threshold is
often a **masking artifact** of an underlying higher-order edge.

**Always filter by `confidence > 0.7` (bootstrap retention frequency),
not by `weight`.** The Studio canvas defaults to this slider. Inferred
edges with `possible_masking_artifact = true` and `confidence < 0.7`
should be dropped; edges with `possible_masking_artifact = true` and
`confidence > 0.7` deserve a manual look (might be a real pairwise
edge that happens to overlap a triadic group).

## Ground-truth validation

Phase 15b validates against EATH-generated synthetic narratives with
planted hyperedges (Mancastroppa, Cencetti, Barrat — arXiv:2507.01124).
Load-bearing test:
`reconstruct_tests::test_reconstruction_recovers_planted_eath_structure_auroc_gt_0_85`
achieves AUROC = 0.852 (> 0.85 threshold).

For your own corpus, compute AUROC by:
1. Run reconstruction → get `Vec<InferredHyperedge>`.
2. Cross-reference each inferred edge's member set against ingested
   coordination ground truth (declared groups from DISARM, OSINT,
   etc.).
3. Sweep the confidence threshold; for each threshold compute TPR/FPR
   over the candidate edge set; integrate the ROC curve.

If AUROC < 0.85 on a real corpus, the escalation path is documented in
`docs/EATH_sprint_extension.md` Phase 15b Q5 — typical fixes: increase
sliding window, use a smoother derivative estimator (SavitzkyGolay
window 7+), tighten Pearson pre-filter (ρ_min ≥ 0.3), bump
`bootstrap_k`.

## Common idioms

**"Find hidden coordination in my disinfo corpus":**
```sql
INFER HYPERGRAPH FROM DYNAMICS FOR "disinfo-tg-q1-2026"
```

**"Compare reconstruction to declared groups":**
```sql
-- Materialize first.
-- Then via REST/MCP filter inferred situations and cross-reference:
MATCH (s:Situation) WHERE s.narrative_id = "disinfo-tg-q1-2026-recon"
    AND s.confidence > 0.7
RETURN s
```

**"Use a non-default observation source (sentiment)":**
```sql
INFER HYPERGRAPH FROM DYNAMICS FOR "n1"
    USING OBSERVATION 'sentiment_mean'
-- Note: requires Situation.properties["sentiment"] populated; otherwise
-- the engine returns InferenceError("PrerequisiteMissing").
```

**"Tune for a sparse corpus":**
```
POST /inference/hypergraph-reconstruction
  { "narrative_id": "n1",
    "params": { "max_order": 2,         -- skip triadic+ to save cost
                "pearson_filter_threshold": 0.3,  -- aggressive pre-filter
                "bootstrap_k": 25 } }            -- more bootstrap stability
```

## When NOT to use reconstruction

- **Don't use to "fill in" missing situations.** Reconstruction infers
  group structure (which entities co-act), not narrative content
  (what they do). Use the writer or the `MissingEventPrediction`
  inference for the latter.
- **Don't use on narratives with < 10×N timesteps.** The engine emits a
  `T < 10·N` warning; results below this threshold are underdetermined
  and shouldn't drive analyst decisions.
- **Don't materialize without filtering.** Default threshold (0.7) is
  load-bearing for separating real edges from masking artifacts.
- **Don't conflate with synth generation.** Synth generates synthetic
  hyperedges from a calibrated null; reconstruction recovers latent
  hyperedges from observed dynamics. The provenance flags
  (`Synthetic { model, run_id }` vs `Reconstructed { source_narrative_id,
  job_id }`) keep them disjoint at the data layer; keep them disjoint
  in your reasoning too.

## See also

- `tensa-synth` skill — the forward (generative) problem.
- `docs/synth_reconstruction_algorithm.md` — full algorithm spec, §13.7
  for the analyst-workflow note.
- `docs/EATH_sprint_extension.md` Phase 15c — TENSA-surface integration
  scope.
