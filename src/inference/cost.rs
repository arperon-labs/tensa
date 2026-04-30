//! Job cost estimation for inference scheduling.
//!
//! Provides rough execution time estimates based on the job type
//! and the size of the data involved (situation/entity counts).
//! Used by the planner and API to inform callers.

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::types::{EntityType, InferenceJobType, NarrativeLevel};

use super::types::InferenceJob;

/// Base costs in milliseconds per job type (for small inputs).
const BASE_CAUSAL_MS: u64 = 2000;
const BASE_GAME_MS: u64 = 500;
const BASE_MOTIVATION_MS: u64 = 1500;
const BASE_COUNTERFACTUAL_MS: u64 = 1000;
const BASE_MISSING_LINKS_MS: u64 = 3000;
const BASE_ANOMALY_MS: u64 = 2000;
// Phase 3
const BASE_PATTERN_MINING_MS: u64 = 5000;
const BASE_ARC_CLASSIFICATION_MS: u64 = 1000;
const BASE_MISSING_EVENT_MS: u64 = 4000;
const _PER_NARRATIVE_MS: u64 = 500;

/// Per-situation cost multiplier in milliseconds.
const PER_SITUATION_MS: u64 = 50;
/// Per-entity cost multiplier for motivation inference.
const PER_ENTITY_TRAJECTORY_MS: u64 = 200;

/// Estimate execution cost for a job in milliseconds.
pub fn estimate_cost(job: &InferenceJob, hypergraph: &Hypergraph) -> Result<u64> {
    let base = match &job.job_type {
        InferenceJobType::CausalDiscovery => {
            let situations = hypergraph
                .list_situations_by_level(NarrativeLevel::Scene)?
                .len() as u64;
            let total_situations = situations.max(
                hypergraph
                    .list_situations_by_level(NarrativeLevel::Event)?
                    .len() as u64,
            );
            BASE_CAUSAL_MS + total_situations * PER_SITUATION_MS
        }
        InferenceJobType::GameClassification => {
            // Cost depends on participants in the target situation
            BASE_GAME_MS
        }
        InferenceJobType::MotivationInference => {
            let actors = hypergraph.list_entities_by_type(&EntityType::Actor)?.len() as u64;
            BASE_MOTIVATION_MS + actors * PER_ENTITY_TRAJECTORY_MS
        }
        InferenceJobType::Counterfactual => {
            // Beam search: cost depends on depth and causal chain length
            let depth = job
                .parameters
                .get("depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(20);
            BASE_COUNTERFACTUAL_MS + depth * PER_SITUATION_MS
        }
        InferenceJobType::MissingLinks => {
            let situations = hypergraph
                .list_situations_by_level(NarrativeLevel::Scene)?
                .len() as u64;
            BASE_MISSING_LINKS_MS + situations * PER_SITUATION_MS
        }
        InferenceJobType::AnomalyDetection => {
            let actors = hypergraph.list_entities_by_type(&EntityType::Actor)?.len() as u64;
            BASE_ANOMALY_MS + actors * PER_ENTITY_TRAJECTORY_MS
        }
        InferenceJobType::PatternMining => {
            let situations = hypergraph
                .list_situations_by_level(NarrativeLevel::Scene)?
                .len() as u64;
            BASE_PATTERN_MINING_MS + situations * PER_SITUATION_MS
        }
        InferenceJobType::ArcClassification => BASE_ARC_CLASSIFICATION_MS,
        InferenceJobType::ActorArcClassification => BASE_ARC_CLASSIFICATION_MS * 2,
        InferenceJobType::MissingEventPrediction => {
            let situations = hypergraph
                .list_situations_by_level(NarrativeLevel::Scene)?
                .len() as u64;
            BASE_MISSING_EVENT_MS + situations * PER_SITUATION_MS
        }
        // Phase 4: Analysis engines
        InferenceJobType::CentralityAnalysis => 5000,
        InferenceJobType::EntropyAnalysis => 3000,
        InferenceJobType::BeliefModeling => 4000,
        InferenceJobType::EvidenceCombination => 1000,
        InferenceJobType::ArgumentationAnalysis => 2000,
        InferenceJobType::ContagionAnalysis => 3000,
        // Narrative Fingerprint (Stylometry)
        InferenceJobType::StyleProfile => 5000,
        InferenceJobType::StyleComparison => 4000,
        InferenceJobType::StyleAnomaly => 6000,
        InferenceJobType::AuthorshipVerification => 8000,
        InferenceJobType::TCGAnomaly => 5000,
        InferenceJobType::NextEvent => 3000,
        InferenceJobType::TemporalILP => 5000,
        InferenceJobType::MeanFieldGame => 4000,
        InferenceJobType::ProbabilisticSoftLogic => 3000,
        InferenceJobType::TrajectoryEmbedding => 5000,
        InferenceJobType::NarrativeSimulation => 30000,
        // Sprint 1: Core graph centrality
        InferenceJobType::PageRank => 5000,
        InferenceJobType::EigenvectorCentrality => 5000,
        InferenceJobType::HarmonicCentrality => 5000,
        InferenceJobType::HITS => 5000,
        // Sprint 2: Topology & community
        InferenceJobType::Topology => 3000,
        InferenceJobType::LabelPropagation => 2000,
        InferenceJobType::KCore => 3000,
        // Sprint 7: Narrative-native
        InferenceJobType::TemporalPageRank => 6000,
        InferenceJobType::CausalInfluence => 5000,
        InferenceJobType::InfoBottleneck => 4000,
        InferenceJobType::Assortativity => 2000,
        // Sprint 8: Temporal patterns
        InferenceJobType::TemporalMotifs => 6000,
        InferenceJobType::FactionEvolution => 8000,
        // Sprint 11: Graph embeddings & network inference
        InferenceJobType::FastRP => 5000,
        InferenceJobType::Node2Vec => 10000,
        InferenceJobType::NetworkInference => 15000,
        // Disinfo Sprint D1: lightweight wrappers around synchronous compute fns.
        InferenceJobType::BehavioralFingerprint => 200,
        InferenceJobType::DisinfoFingerprint => 500,
        // Disinfo Sprint D2: SMIR re-runs the contagion simulator and
        // critical-spreader sweep — heavier but bounded by narrative size.
        InferenceJobType::SpreadVelocity => 4000,
        InferenceJobType::SpreadIntervention => 5000,
        // Disinfo Sprint D3: behavioral similarity network + density bootstrap.
        // Cost is O(N²) pair comparisons + bootstrap_iter × size_sweep.
        InferenceJobType::CibDetection => 6000,
        InferenceJobType::Superspreaders => 3000,
        // Disinfo Sprint D4: Claims pipeline — origin trace walks the causal
        // chain (bounded), match scans stored fact-checks (light).
        InferenceJobType::ClaimOrigin => 4000,
        InferenceJobType::ClaimMatch => 2000,
        // Disinfo Sprint D5: archetype classification + DS fusion.
        InferenceJobType::ArchetypeClassification => 200,
        InferenceJobType::DisinfoAssessment => 100,
        // Sprint D9: Narrative architecture
        InferenceJobType::CommitmentDetection => 5000,
        InferenceJobType::FabulaExtraction => 3000,
        InferenceJobType::SjuzetExtraction => 3000,
        InferenceJobType::DramaticIrony => 5000,
        InferenceJobType::Focalization => 3000,
        InferenceJobType::CharacterArc => 5000,
        InferenceJobType::SubplotDetection => 5000,
        InferenceJobType::SceneSequel => 3000,
        InferenceJobType::SjuzetReordering => 4000,
        // Registry-only variants — the actual analyses run via dedicated
        // endpoints (POST /narratives/:id/communities/summarize and the
        // diagnose_narrative MCP tool) outside the worker pool, so the
        // cost here is a placeholder for any future bulk-runner submission.
        InferenceJobType::CommunitySummary => 5000,
        InferenceJobType::NarrativeDiagnose => 4000,
        // Sprint D12: Adversarial narrative wargaming
        InferenceJobType::AdversaryPolicy => 2000,
        InferenceJobType::CognitiveHierarchy => 1000,
        InferenceJobType::WargameSimulation => 30000,
        InferenceJobType::RewardFingerprint => 5000,
        InferenceJobType::CounterNarrative => 30000,
        InferenceJobType::Retrodiction => 60000,
        // ~3 LLM calls per loop, conservative ceiling.
        InferenceJobType::ChapterGenerationFitness => 90000,
        // EATH synthetic-generation sprint — Phase 4 wires the engines for
        // SurrogateCalibration + SurrogateGeneration. Significance + contagion
        // significance still flat-rate (Phases 7 / 7b own those costs).
        //
        // Calibration: linear in (entities + situations) of the source.
        // Computing the size needs a list scan; we cap at 100 ms / record so
        // the figure is small enough to be useful as a scheduling hint
        // without spending real time computing the cost itself.
        InferenceJobType::SurrogateCalibration { narrative_id, .. } => {
            estimate_calibration_cost(hypergraph, narrative_id.as_str())
        }
        // Generation: roughly num_entities × num_steps / 10 (rough ms count
        // — gen writes one Situation per group per step, dominated by KV
        // round-trips). When params_override carries a fitted EathParams
        // we read num_entities from it; otherwise we look up the source
        // narrative size; otherwise default 100. num_steps reads from
        // job.parameters.num_steps if set, else the engine default 100.
        InferenceJobType::SurrogateGeneration {
            source_narrative_id,
            params_override,
            ..
        } => estimate_generation_cost(
            hypergraph,
            source_narrative_id.as_deref(),
            params_override.as_ref(),
            &job.parameters,
        ),
        InferenceJobType::SurrogateSignificance {
            narrative_id,
            k,
            model,
            ..
        } => estimate_significance_cost(hypergraph, narrative_id, model, *k),
        InferenceJobType::SurrogateContagionSignificance {
            narrative_id,
            k,
            model,
            contagion_params,
        } => estimate_contagion_significance_cost(
            hypergraph,
            narrative_id,
            model,
            *k,
            contagion_params,
        ),
        // EATH Phase 9: hybrid generation. Same per-step / per-entity cost
        // shape as single-source generation — each step samples one source
        // and recruits one group. Use the same 100×100 default basis as
        // SurrogateGeneration when num_steps is None / source is unknown.
        InferenceJobType::SurrogateHybridGeneration { num_steps, .. } => {
            let n = num_steps.unwrap_or(100) as u64;
            // Conservative: ~1ms per step.
            (n.max(1) * 100).max(1_000)
        }
        // EATH Extension Phase 13c: dual-null-model significance. Cost is
        // sum-of-models because each model's K-loop is independent. We use
        // the same per-model formula as `estimate_significance_cost` so
        // adding a third model just adds its slice to the total. NuDHy +
        // EATH default → 2× the single-model figure.
        InferenceJobType::SurrogateDualSignificance {
            narrative_id,
            k_per_model,
            models,
            ..
        } => {
            let model_list: Vec<String> = if models.is_empty() {
                vec!["eath".into(), "nudhy".into()]
            } else {
                models.clone()
            };
            model_list
                .iter()
                .map(|m| estimate_significance_cost(hypergraph, narrative_id, m, *k_per_model))
                .sum::<u64>()
                .max(1_000)
        }
        // EATH Extension Phase 14: bistability significance. Each (model, K)
        // sample runs a full forward-backward sweep over `num_betas` × 2
        // branches × `replicates_per_beta` simulations. We use the
        // single-model significance cost as a baseline scaled by an estimate
        // of the per-sample sweep workload.
        // EATH Extension Phase 15b: SINDy hypergraph reconstruction.
        // Cost ~ T × library_size × N × K_bootstrap. We can't size the
        // library precisely without the Pearson pre-filter result, so we
        // use a reasonable upper bound: at N entities and max_order=3
        // with the default Pearson filter retaining ~10% of pairs, the
        // effective library is ≈ N + 0.1·C(N,2) + 0.01·C(N,3) terms. T
        // and K_bootstrap come straight from `params` when present.
        InferenceJobType::HypergraphReconstruction {
            narrative_id,
            params,
        } => estimate_reconstruction_cost(hypergraph, narrative_id, params),
        InferenceJobType::SurrogateBistabilitySignificance {
            narrative_id,
            k,
            models,
            params,
        } => {
            let model_list: Vec<String> = if models.is_empty() {
                vec!["eath".into(), "nudhy".into()]
            } else {
                models.clone()
            };
            // Best-effort sweep workload multiplier: num_betas × replicates × 2.
            let sweep_workload: u64 = params
                .get("beta_0_range")
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(2))
                .and_then(|v| v.as_u64())
                .unwrap_or(20)
                .saturating_mul(
                    params
                        .get("replicates_per_beta")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10),
                )
                .saturating_mul(2)
                .max(20);
            let per_model: u64 = model_list
                .iter()
                .map(|m| {
                    estimate_significance_cost(hypergraph, narrative_id, m, *k)
                        .saturating_mul(sweep_workload)
                        / 200
                })
                .sum();
            per_model.max(1_000)
        }
        // EATH Extension Phase 16c: opinion-dynamics significance.
        // Per-K cost is dominated by max_steps × N_entities. We estimate a
        // floor based on K samples × per-model and scale linearly with K.
        InferenceJobType::SurrogateOpinionSignificance {
            narrative_id,
            k,
            models,
            ..
        } => {
            let model_list: Vec<String> = if models.is_empty() {
                vec!["eath".into(), "nudhy".into()]
            } else {
                models.clone()
            };
            let per_model: u64 = model_list
                .iter()
                .map(|m| {
                    // Approximate: each opinion-dynamics simulation is ~50 ms
                    // per surrogate at MVP scales (Phase 16b benchmark);
                    // calibration adds the per-model significance baseline.
                    estimate_significance_cost(hypergraph, narrative_id, m, *k) / 5
                })
                .sum();
            per_model.max(1_000)
        }
        // ── Fuzzy Logic Sprint (Phase 0 stubs) ─────────────────────────────
        // Seven variants share a flat scheduling floor until their owning
        // phases (1 → 10) land real engines. Phase 0 contract: every arm
        // compiles, the cost estimator stays exhaustive.
        InferenceJobType::FuzzyAggregate { .. } => 1_000,
        InferenceJobType::FuzzyAllenGradation { .. } => 2_000,
        InferenceJobType::FuzzyQuantifierEvaluate { .. } => 2_000,
        InferenceJobType::FuzzySyllogismVerify { .. } => 3_000,
        InferenceJobType::FuzzyFcaLattice { .. } => 10_000,
        InferenceJobType::FuzzyRuleEvaluate { .. } => 1_000,
        InferenceJobType::FuzzyHybridInference { .. } => 5_000,
    };

    Ok(base)
}

// ── EATH cost helpers ────────────────────────────────────────────────────────

/// Per-record cost weight for surrogate calibration. Single-pass walk over
/// entities + situations is genuinely O(N + S) but each record is touched
/// briefly; 1 ms per record is a comfortable upper bound.
const CAL_PER_RECORD_MS: u64 = 1;
/// Floor cost when narrative is unknown / empty — the fidelity report's K
/// synthetic samples dominate even an empty source.
const CAL_FLOOR_MS: u64 = 10_000;
/// Ceiling so a million-record narrative doesn't return an unreasonable
/// estimate. Cost is just a scheduling hint.
const CAL_CEILING_MS: u64 = 600_000;
/// Generation throughput hint: ms per (entity × step). Tuned so a
/// 100-entity × 100-step run rounds to ~1 second.
const GEN_MS_PER_ENTITY_STEP_DENOM: u64 = 10_000;
/// Floor cost for generation — even tiny runs pay queue overhead.
const GEN_FLOOR_MS: u64 = 1_000;
/// Default fallback when no source narrative + no override params.
const GEN_FALLBACK_ENTITIES: u64 = 100;
/// Default fallback when no num_steps is provided.
const GEN_FALLBACK_STEPS: u64 = 100;

/// Compute calibration cost from the source narrative size.
///
/// Walks the per-narrative entity + situation indexes (cheap prefix scans).
/// On any error, returns the floor — cost is just a scheduling hint, never
/// load-bearing for correctness.
fn estimate_calibration_cost(hypergraph: &Hypergraph, narrative_id: &str) -> u64 {
    if narrative_id.is_empty() {
        return CAL_FLOOR_MS;
    }
    let n_entities = hypergraph
        .list_entities_by_narrative(narrative_id)
        .map(|v| v.len() as u64)
        .unwrap_or(0);
    let n_situations = hypergraph
        .list_situations_by_narrative(narrative_id)
        .map(|v| v.len() as u64)
        .unwrap_or(0);
    let raw = (n_entities + n_situations).saturating_mul(CAL_PER_RECORD_MS);
    raw.clamp(CAL_FLOOR_MS, CAL_CEILING_MS)
}

/// Compute generation cost from `num_entities × num_steps`.
///
/// Resolution order for `num_entities`:
/// 1. `params_override.num_entities` (when provided),
/// 2. source narrative entity count,
/// 3. fallback constant.
///
/// Resolution order for `num_steps`:
/// 1. `job.parameters.num_steps` (matches the engine's own resolution),
/// 2. fallback constant.
fn estimate_generation_cost(
    hypergraph: &Hypergraph,
    source_narrative_id: Option<&str>,
    params_override: Option<&serde_json::Value>,
    job_parameters: &serde_json::Value,
) -> u64 {
    let num_entities = params_override
        .and_then(|p| p.get("num_entities"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            source_narrative_id.and_then(|nid| {
                hypergraph
                    .list_entities_by_narrative(nid)
                    .ok()
                    .map(|v| v.len() as u64)
            })
        })
        .unwrap_or(GEN_FALLBACK_ENTITIES);
    let num_steps = job_parameters
        .get("num_steps")
        .and_then(|v| v.as_u64())
        .unwrap_or(GEN_FALLBACK_STEPS);
    let raw = num_entities.saturating_mul(num_steps) / GEN_MS_PER_ENTITY_STEP_DENOM;
    raw.max(GEN_FLOOR_MS)
}

/// Phase 7: SurrogateSignificance cost. Roughly
/// `num_entities × num_steps × k / SIG_THROUGHPUT_DENOM` (seconds estimate).
///
/// Resolution order:
/// * `num_entities` and `num_steps` come from `load_params(narrative_id, model)`
///   when calibrated params exist; otherwise fall back to (100, 200).
/// * `k` clamped at [`crate::synth::significance::K_MAX`] to mirror engine behavior.
fn estimate_significance_cost(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    model: &str,
    k: u16,
) -> u64 {
    let (num_entities, num_steps) = if narrative_id.is_empty() {
        (SIG_FALLBACK_ENTITIES, SIG_FALLBACK_STEPS)
    } else {
        crate::synth::calibrate::load_params(hypergraph.store(), narrative_id, model)
            .ok()
            .flatten()
            .map(|p| {
                let n = p.num_entities.max(p.a_t_distribution.len()) as u64;
                (n.max(1), SIG_FALLBACK_STEPS)
            })
            .unwrap_or((SIG_FALLBACK_ENTITIES, SIG_FALLBACK_STEPS))
    };
    let k_eff = (k as u64).clamp(1, crate::synth::significance::K_MAX as u64);
    let raw = num_entities
        .saturating_mul(num_steps)
        .saturating_mul(k_eff)
        / SIG_THROUGHPUT_DENOM;
    raw.max(SIG_FLOOR_MS)
}

/// Phase 7b: SurrogateContagionSignificance cost. Higher-order SIR is
/// rougher than the temporal-motifs/communities/patterns metrics (per-step,
/// per-entity, per-hyperedge enumeration), so the throughput denominator is
/// smaller than [`SIG_THROUGHPUT_DENOM`].
///
/// Formula: `num_entities × max_steps × k / CONTAGION_SIG_THROUGHPUT_DENOM`,
/// floored at [`SIG_FLOOR_MS`]. `max_steps` reads from the contagion params
/// blob when present, falling back to a sensible default.
fn estimate_contagion_significance_cost(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    model: &str,
    k: u16,
    contagion_params: &serde_json::Value,
) -> u64 {
    let num_entities = if narrative_id.is_empty() {
        SIG_FALLBACK_ENTITIES
    } else {
        crate::synth::calibrate::load_params(hypergraph.store(), narrative_id, model)
            .ok()
            .flatten()
            .map(|p| (p.num_entities.max(p.a_t_distribution.len()) as u64).max(1))
            .unwrap_or(SIG_FALLBACK_ENTITIES)
    };
    let max_steps = contagion_params
        .get("max_steps")
        .and_then(|v| v.as_u64())
        .unwrap_or(CONTAGION_FALLBACK_STEPS);
    let k_eff = (k as u64).clamp(1, crate::synth::significance::K_MAX as u64);
    let raw = num_entities
        .saturating_mul(max_steps)
        .saturating_mul(k_eff)
        / CONTAGION_SIG_THROUGHPUT_DENOM;
    raw.max(SIG_FLOOR_MS)
}

// ── EATH Phase 15b: hypergraph reconstruction cost helpers ─────────────────────

/// Cost denominator for SINDy reconstruction. Tuned so a 50-entity × 1000-step
/// run with K=10 bootstraps lands near the 30s performance target.
const RECON_THROUGHPUT_DENOM: u64 = 50_000;
/// Minimum cost for any reconstruction job — even tiny narratives pay queue
/// overhead and Pearson-matrix construction.
const RECON_FLOOR_MS: u64 = 2_000;
/// Default narrative size when we can't read the source (empty narrative_id,
/// store unreachable, etc.).
const RECON_FALLBACK_ENTITIES: u64 = 50;
/// Default time-axis length when params don't carry an explicit hint.
const RECON_FALLBACK_T: u64 = 500;
/// Default K_bootstrap when params don't carry it.
const RECON_FALLBACK_K: u64 = 10;
/// Default max_order when params don't carry it.
const RECON_FALLBACK_MAX_ORDER: u64 = 3;
/// Default Pearson filter retention (≈10% of candidate pairs survive).
const RECON_PEARSON_RETENTION_PCT: u64 = 10;

/// Estimate the SINDy hypergraph reconstruction job cost.
///
/// We approximate the dominant work (library × LASSO × bootstrap) without
/// actually building the matrices. Best-effort: any error reading the
/// narrative's entity count falls back to the constants above.
fn estimate_reconstruction_cost(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    params: &serde_json::Value,
) -> u64 {
    let n_entities = if narrative_id.is_empty() {
        RECON_FALLBACK_ENTITIES
    } else {
        hypergraph
            .list_entities_by_narrative(narrative_id)
            .map(|v| (v.len() as u64).max(2))
            .unwrap_or(RECON_FALLBACK_ENTITIES)
    };
    let t_estimate = params
        .get("time_resolution_seconds")
        .and_then(|v| v.as_u64())
        .map(|res| {
            params
                .get("window_seconds")
                .and_then(|w| w.as_u64())
                .map(|w| w.saturating_mul(1024) / res.max(1))
        })
        .flatten()
        .unwrap_or(RECON_FALLBACK_T);
    let k_boot = params
        .get("bootstrap_k")
        .and_then(|v| v.as_u64())
        .unwrap_or(RECON_FALLBACK_K);
    let max_order = params
        .get("max_order")
        .and_then(|v| v.as_u64())
        .unwrap_or(RECON_FALLBACK_MAX_ORDER)
        .clamp(1, 4);

    // Effective library size estimate: N + retained_pairs + retained_triples.
    let pairs = n_entities.saturating_mul(n_entities.saturating_sub(1)) / 2;
    let triples = if max_order >= 3 {
        n_entities
            .saturating_mul(n_entities.saturating_sub(1))
            .saturating_mul(n_entities.saturating_sub(2))
            / 6
    } else {
        0
    };
    let retained = pairs * RECON_PEARSON_RETENTION_PCT / 100
        + triples * RECON_PEARSON_RETENTION_PCT * RECON_PEARSON_RETENTION_PCT / 10_000;
    let lib_size = n_entities + retained;

    let raw = t_estimate
        .saturating_mul(lib_size)
        .saturating_mul(n_entities)
        .saturating_mul(k_boot.max(1))
        / RECON_THROUGHPUT_DENOM;
    raw.max(RECON_FLOOR_MS)
}

/// Phase 7b throughput denominator. Half of [`SIG_THROUGHPUT_DENOM`] —
/// per-step transmission enumeration is roughly 2× the cost of metric
/// passes that walk only entities/edges once.
const CONTAGION_SIG_THROUGHPUT_DENOM: u64 = 50_000;
/// Default `max_steps` when the contagion params blob omits it.
const CONTAGION_FALLBACK_STEPS: u64 = 100;

/// Significance throughput denominator. 100k yields ~1 second per
/// 100×100×K=100 sample run, matching the design doc §5 estimates.
const SIG_THROUGHPUT_DENOM: u64 = 100_000;
/// Floor cost for significance — even tiny K=1 runs pay queue overhead +
/// metric-on-source computation.
const SIG_FLOOR_MS: u64 = 1_000;
/// Default num_entities when no calibrated params exist.
const SIG_FALLBACK_ENTITIES: u64 = 100;
/// Default num_steps used in the K-loop generator (matches
/// `synth::significance::adapters::generate_one_sample`).
const SIG_FALLBACK_STEPS: u64 = 200;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::sync::Arc;
    use uuid::Uuid;

    fn setup() -> Hypergraph {
        let store = Arc::new(MemoryStore::new());
        Hypergraph::new(store)
    }

    fn make_job(job_type: InferenceJobType) -> InferenceJob {
        InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 0,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    #[test]
    fn test_causal_base_cost() {
        let hg = setup();
        let job = make_job(InferenceJobType::CausalDiscovery);
        let cost = estimate_cost(&job, &hg).unwrap();
        assert!(cost >= BASE_CAUSAL_MS);
    }

    #[test]
    fn test_game_base_cost() {
        let hg = setup();
        let job = make_job(InferenceJobType::GameClassification);
        let cost = estimate_cost(&job, &hg).unwrap();
        assert_eq!(cost, BASE_GAME_MS);
    }

    #[test]
    fn test_motivation_cost_scales_with_actors() {
        let hg = setup();
        // Add some actors
        for i in 0..3 {
            let entity = Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": format!("Actor {}", i)}),
                beliefs: None,
                embedding: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            hg.create_entity(entity).unwrap();
        }

        let job = make_job(InferenceJobType::MotivationInference);
        let cost = estimate_cost(&job, &hg).unwrap();
        assert_eq!(cost, BASE_MOTIVATION_MS + 3 * PER_ENTITY_TRAJECTORY_MS);
    }

    #[test]
    fn test_counterfactual_cost_uses_depth_param() {
        let hg = setup();
        let mut job = make_job(InferenceJobType::Counterfactual);
        job.parameters = serde_json::json!({"depth": 10});
        let cost = estimate_cost(&job, &hg).unwrap();
        assert_eq!(cost, BASE_COUNTERFACTUAL_MS + 10 * PER_SITUATION_MS);
    }

    #[test]
    fn test_all_job_types_return_positive_cost() {
        let hg = setup();
        let types = vec![
            InferenceJobType::CausalDiscovery,
            InferenceJobType::GameClassification,
            InferenceJobType::MotivationInference,
            InferenceJobType::Counterfactual,
            InferenceJobType::MissingLinks,
            InferenceJobType::AnomalyDetection,
            InferenceJobType::PatternMining,
            InferenceJobType::ArcClassification,
            InferenceJobType::MissingEventPrediction,
            InferenceJobType::CentralityAnalysis,
            InferenceJobType::EntropyAnalysis,
            InferenceJobType::BeliefModeling,
            InferenceJobType::EvidenceCombination,
            InferenceJobType::ArgumentationAnalysis,
            InferenceJobType::ContagionAnalysis,
            InferenceJobType::TemporalILP,
            InferenceJobType::MeanFieldGame,
            InferenceJobType::ProbabilisticSoftLogic,
            InferenceJobType::TrajectoryEmbedding,
            InferenceJobType::NarrativeSimulation,
            InferenceJobType::PageRank,
            InferenceJobType::EigenvectorCentrality,
            InferenceJobType::HarmonicCentrality,
            InferenceJobType::HITS,
            InferenceJobType::Topology,
            InferenceJobType::LabelPropagation,
            InferenceJobType::KCore,
            InferenceJobType::TemporalPageRank,
            InferenceJobType::CausalInfluence,
            InferenceJobType::InfoBottleneck,
            InferenceJobType::Assortativity,
            InferenceJobType::TemporalMotifs,
            InferenceJobType::FactionEvolution,
            // Sprint D12: Adversarial narrative wargaming
            InferenceJobType::AdversaryPolicy,
            InferenceJobType::CognitiveHierarchy,
            InferenceJobType::WargameSimulation,
            InferenceJobType::RewardFingerprint,
            InferenceJobType::CounterNarrative,
            InferenceJobType::Retrodiction,
            // EATH synthetic-generation sprint
            InferenceJobType::SurrogateCalibration {
                narrative_id: "n".into(),
                model: "eath".into(),
            },
            InferenceJobType::SurrogateGeneration {
                source_narrative_id: None,
                output_narrative_id: "out".into(),
                model: "eath".into(),
                params_override: None,
                seed_override: None,
            },
            InferenceJobType::SurrogateSignificance {
                narrative_id: "n".into(),
                metric_kind: "motifs".into(),
                k: 20,
                model: "eath".into(),
            },
            InferenceJobType::SurrogateContagionSignificance {
                narrative_id: "n".into(),
                k: 20,
                model: "eath".into(),
                contagion_params: serde_json::json!({}),
            },
            InferenceJobType::SurrogateHybridGeneration {
                components: serde_json::json!([]),
                output_narrative_id: "out".into(),
                seed_override: None,
                num_steps: Some(100),
            },
            InferenceJobType::SurrogateDualSignificance {
                narrative_id: "n".into(),
                metric: "patterns".into(),
                k_per_model: 20,
                models: vec!["eath".into(), "nudhy".into()],
            },
            InferenceJobType::SurrogateBistabilitySignificance {
                narrative_id: "n".into(),
                params: serde_json::json!({"beta_0_range": [0.0, 1.0, 8]}),
                k: 5,
                models: vec!["eath".into(), "nudhy".into()],
            },
            // EATH Phase 15b: SINDy hypergraph reconstruction.
            InferenceJobType::HypergraphReconstruction {
                narrative_id: "n".into(),
                params: serde_json::json!({"max_order": 3}),
            },
            // EATH Extension Phase 16c: opinion-dynamics significance.
            InferenceJobType::SurrogateOpinionSignificance {
                narrative_id: "n".into(),
                params: serde_json::json!({"confidence_bound": 0.3}),
                k: 5,
                models: vec!["eath".into()],
            },
            // Fuzzy Logic Sprint (Phase 0 stubs).
            InferenceJobType::FuzzyAggregate {
                narrative_id: "n".into(),
                target_id: uuid::Uuid::now_v7(),
                config: serde_json::json!({"tnorm": "godel", "aggregator": "mean"}),
            },
            InferenceJobType::FuzzyAllenGradation {
                narrative_id: "n".into(),
                situation_pair: (uuid::Uuid::now_v7(), uuid::Uuid::now_v7()),
            },
            InferenceJobType::FuzzyQuantifierEvaluate {
                narrative_id: "n".into(),
                predicate: "is_partisan".into(),
                quantifier: "most".into(),
            },
            InferenceJobType::FuzzySyllogismVerify {
                narrative_id: "n".into(),
                premises: vec!["most leaders are partisan".into()],
                conclusion: "some leaders are partisan".into(),
            },
            InferenceJobType::FuzzyFcaLattice {
                narrative_id: "n".into(),
                attribute_filter: None,
            },
            InferenceJobType::FuzzyRuleEvaluate {
                narrative_id: "n".into(),
                rule_id: "r1".into(),
                entity_id: None,
            },
            InferenceJobType::FuzzyHybridInference {
                narrative_id: "n".into(),
                config: serde_json::json!({"event": {}, "dist": {}}),
            },
        ];
        for jt in types {
            let job = make_job(jt);
            let cost = estimate_cost(&job, &hg).unwrap();
            assert!(cost > 0);
        }
    }
}
