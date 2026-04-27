---
name: tensa-opinion-dynamics
description: Bounded-confidence opinion dynamics on hypergraphs (BCM + Deffuant). Activate when the user wants to know whether a narrative converges or fragments, find critical confidence thresholds, or detect echo chambers.
---

# TENSA Opinion Dynamics Assistant

Activate this skill when the user wants to:

- **Predict whether a population reaches consensus or fragments**
  given the structure of a narrative's coordination edges and a
  configurable confidence threshold.
- **Find the critical confidence bound** at which a narrative
  transitions from rapid global consensus to persistent fragmentation
  (the Hickok 2022 σ²-spike).
- **Detect echo chambers** — clusters whose final opinions strongly
  align with their pre-existing community labels (Label Propagation
  output at `an/lp/`).
- **Score interventions in wargames** by how much they shift
  aggregate opinion toward a target value (Phase 16c
  `RewardFunction::OpinionShift`).

This is a **separate problem** from synth generation (forward sampling
of synthetic hypergraphs) and from reconstruction (recovering latent
group structure from observed dynamics). Opinion dynamics treats the
hypergraph as fixed and watches scalar opinions x_i ∈ [0, 1] evolve
under local update rules.

## Algorithm variants

Two MVP variants share the same selection / convergence machinery and
differ only in the per-edge update rule:

- **PairwiseWithin** (default) — Hickok, Kureh, Brooks, Feng, Porter,
  *A Bounded-Confidence Model of Opinion Dynamics on Hypergraphs*,
  SIAM J. Appl. Dyn. Syst. **21**, 1 (2022). Lifts Deffuant 2000
  dyadic BCM to higher-order edges by applying the pairwise update to
  every ordered pair within the selected hyperedge in canonical (sorted-UUID)
  order — Gauss-Seidel, with updates immediately visible to subsequent
  pairs in the same edge. Reduces to dyadic Deffuant on size-2 edges.
- **GroupMean** — Schawe & Hernández, *Higher order interactions
  destroy phase transitions in Deffuant opinion dynamics model*,
  Commun. Phys. **5**, 32 (2022). All-or-nothing group update: when
  the spread within the selected edge is below the (size-scaled)
  confidence bound, every member moves toward the group mean.
  Otherwise no update. Produces a *smooth crossover* (no sharp phase
  transition).

## Key parameters

```rust
OpinionDynamicsParams {
    model: BcmVariant,                    // PairwiseWithin (default) | GroupMean
    confidence_bound: f32,                // c ∈ (0, 1); default 0.3
    confidence_size_scaling: Option<...>, // Flat (default) | InverseSqrtSize | InverseSize
    convergence_rate: f32,                // μ ∈ (0, 1]; default 0.5 (Deffuant canonical)
    hyperedge_selection: HyperedgeSelection, // UniformRandom (default) | ActivityWeighted | PerStepAll
    initial_opinion_distribution: InitialOpinionDist, // Uniform | Gaussian | Bimodal | Custom
    convergence_tol: f32,                 // ε_conv (1e-4 default)
    convergence_window: usize,            // N_conv consecutive sub-tolerance steps (100 default)
    max_steps: usize,                     // 100k default
    seed: u64,                            // 42 default
}
```

## TensaQL surface

```sql
-- Defaults: variant = PairwiseWithin, c = 0.3, μ = 0.5, initial = Uniform.
INFER OPINION_DYNAMICS(
    confidence_bound := 0.3,
    variant := 'pairwise'
) FOR "narr-1"

-- Tunable form.
INFER OPINION_DYNAMICS(
    confidence_bound := 0.4,
    variant := 'group_mean',
    mu := 0.5,
    initial := 'gaussian'
) FOR "n1"

-- Phase-transition sweep (Hickok §5; distinct from Phase 14 bistability).
INFER OPINION_PHASE_TRANSITION(
    c_start := 0.05,
    c_end := 0.5,
    c_steps := 10
) FOR "narr-1"
```

Both forms execute synchronously (Phase 16b benchmark: 100×10k steps
≈ 21 ms; 1000×100k ≈ 98 ms — no job queue needed at MVP scales).

## REST surface

```
POST /analysis/opinion-dynamics
  body: { narrative_id, params?: OpinionDynamicsParams,
          include_synthetic?: bool }
  → 200 { run_id, report: OpinionDynamicsReport }

POST /analysis/opinion-dynamics/phase-transition-sweep
  body: { narrative_id, c_range: [start, end, steps],
          base_params?: OpinionDynamicsParams }
  → 200 PhaseTransitionReport

POST /synth/opinion-significance     -- only if Phase 13 (NuDHy) shipped
  body: { narrative_id, params?, k?, models? }
  → 201 { job_id, status: "Pending" }   -- queued; result via job_id
```

Each successful run persists at
`opd/report/{narrative_id}/{run_id_v7_BE}` for chronological readback.

Consensus detection can run under any t-norm via `?tnorm=<kind>`
(Fuzzy Sprint, v0.78.0) — the bounded-confidence comparison
`|op_i − op_j| < c` is graded under the selected semantics; default
Gödel preserves pre-sprint behaviour bit-identically.

## MCP tools

- `simulate_opinion_dynamics(narrative_id, params?)` → inline `{ run_id, report }`.
- `simulate_opinion_phase_transition(narrative_id, c_range, base_params?)` → inline `PhaseTransitionReport`.

## Phase transition vs Phase 14 bistability — DON'T CONFLATE

| Observable | Phase 14 Bistability | Phase 16 Phase Transition |
|---|---|---|
| What varies | Transmission rate β | Confidence bound c |
| Observable measured | Final infected prevalence | Time to convergence |
| Phenomenon | Bistable interval where two stable prevalence states coexist | Sharp spike in convergence time near c ≈ σ² |
| Model | SIR higher-order contagion | BCM opinion dynamics |
| Source | Ferraz de Arruda et al. 2023 | Hickok et al. 2022 §5 |
| Phase signal | Hysteresis gap > threshold | Convergence-time spike |

Use Phase 14 (`/analysis/contagion-bistability`) for "is this narrative
in the bistable contagion regime?" Use Phase 16 phase-transition sweep
for "at what confidence threshold does this narrative stop converging
to consensus?"

## Wargame integration (Phase 16c, --features adversarial)

```rust
RewardFunction::OpinionShift {
    target_opinion: 0.2,
    baseline_params: OpinionDynamicsParams::default(),
    post_intervention_params: OpinionDynamicsParams::default(),
    aggregator: OpinionAggregator::Mean | Median | ClusterMass{cluster_idx},
}
```

At evaluation time:
1. Run opinion dynamics on baseline substrate (no intervention).
2. Run opinion dynamics on post-intervention substrate.
3. Reward = `|baseline_agg - target| - |treatment_agg - target|` —
   positive when intervention moved aggregate closer to target.

## Echo chambers — graceful degradation

`OpinionDynamicsReport.echo_chamber_index` requires precomputed
Label-Propagation labels at `an/lp/{narrative_id}/{entity_id}`. When
labels are missing, the report carries `echo_chamber_available =
false` and `echo_chamber_index = 0.0`. The analyst sees the
missing-data signal — no panic, no error.

To populate the labels first, run Label Propagation:
```sql
INFER LABEL_PROPAGATION FOR n:Narrative WHERE n.id = "narr-1" RETURN n
```
then re-run opinion dynamics — `echo_chamber_available` will flip to
true.

## Common idioms

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
-- 1. Compute community labels first.
INFER LABEL_PROPAGATION FOR n:Narrative WHERE n.id = "narr-1" RETURN n
-- 2. Then opinion dynamics — echo_chamber_index becomes meaningful.
INFER OPINION_DYNAMICS( confidence_bound := 0.3, variant := 'pairwise' )
    FOR "narr-1"
-- → report.echo_chamber_index ∈ [0, 1]; closer to 1 = strong alignment
--   between final opinion clusters and pre-existing community labels.
```

## When NOT to use opinion dynamics

- **Don't use as a substitute for SIR contagion** — it answers "do
  agents converge to consensus" not "does information spread". Use
  `/analysis/higher-order-contagion` for spread questions.
- **Don't use on narratives with no size-≥2 hyperedges** — the engine
  errors `InvalidInput` because BCM updates need at least one
  multi-actor situation to operate on.
- **Don't conflate the c-sweep with Phase 14 β-sweep** — see the table
  above. They measure different observables on different phenomena.

## See also

- `tensa` skill — the general TensaQL grammar.
- `docs/opinion_dynamics_algorithm.md` — full algorithm spec.
- `docs/EATH_sprint_extension.md` Phase 16a/16b/16c — sprint history.
