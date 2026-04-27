//! Per-sample helpers for [`super::bistability_significance_engine`].
//!
//! Split out of the engine module to keep both files under the 500-line cap.
//! Public to `super` only — these helpers are not part of the synth module's
//! external surface.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::Utc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::analysis::contagion_bistability::{
    run_bistability_sweep, BistabilityReport, BistabilitySweepParams,
};
use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::store::memory::MemoryStore;
use crate::store::KVStore;
use crate::types::{Entity, MaturityLevel};

use super::eath::EathSurrogate;
use super::emit::{
    filter_synthetic_entities, write_synthetic_situation, EmitContext,
    DEFAULT_SYNTHETIC_CONFIDENCE,
};
use super::nudhy::{run_nudhy_chain, NudhyParams, NudhyState};
use super::surrogate::SurrogateModel;
use super::types::{EathParams, SurrogateParams};

/// Run one EATH-generated synthetic narrative through the bistability sweep
/// and return the resulting report. Hypergraph + store dropped at exit.
pub(super) fn run_one_eath_sample(
    eath: &EathParams,
    base_seed: u64,
    sample_idx: usize,
    params: &BistabilitySweepParams,
) -> Result<BistabilityReport> {
    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let target = Hypergraph::new(store);
    let envelope = SurrogateParams {
        model: "eath".into(),
        params_json: serde_json::to_value(eath)?,
        seed: per_sample_seed(base_seed, sample_idx),
        num_steps: 200,
        label_prefix: "bistab-eath".into(),
    };
    let output_narrative_id = format!("bistab-eath-{sample_idx}");
    let summary = EathSurrogate.generate(&envelope, &target, &output_narrative_id)?;
    run_bistability_sweep(&target, &summary.output_narrative_id, params)
}

/// Run one NuDHy MCMC chain, materialise the resulting hyperedges into an
/// ephemeral hypergraph, then run the bistability sweep on it. Mirrors
/// [`super::dual_significance_engine::run_one_nudhy_sample`] minus metric.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_one_nudhy_sample(
    nudhy_params: &NudhyParams,
    chain_edges: Vec<Vec<Uuid>>,
    fixed_edges: Vec<Vec<Uuid>>,
    chain_seed: u64,
    k_idx: usize,
    source_hg: &Hypergraph,
    source_narrative_id: &str,
    params: &BistabilitySweepParams,
) -> Result<BistabilityReport> {
    let initial_state = NudhyState::from_hyperedges(chain_edges);
    let mut rng = ChaCha8Rng::seed_from_u64(chain_seed);
    let final_state = run_nudhy_chain(initial_state, nudhy_params, &mut rng)?;

    let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
    let target = Hypergraph::new(store);
    let sample_narrative_id = format!("bistab-nudhy-{k_idx}");

    // Copy referenced source entities into the ephemeral store.
    let mut needed: HashSet<Uuid> = HashSet::new();
    for e in final_state.hyperedges.iter().chain(fixed_edges.iter()) {
        needed.extend(e.iter().copied());
    }
    let source_entities = source_hg.list_entities_by_narrative(source_narrative_id)?;
    let source_entities = filter_synthetic_entities(source_entities, false);
    for src in source_entities {
        if !needed.contains(&src.id) {
            continue;
        }
        let now = Utc::now();
        let copy = Entity {
            id: src.id,
            entity_type: src.entity_type,
            properties: src.properties.clone(),
            beliefs: src.beliefs.clone(),
            embedding: src.embedding.clone(),
            maturity: MaturityLevel::Candidate,
            confidence: src.confidence,
            confidence_breakdown: src.confidence_breakdown.clone(),
            provenance: vec![],
            extraction_method: src.extraction_method.clone(),
            narrative_id: Some(sample_narrative_id.clone()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        target.create_entity(copy)?;
    }

    let run_id = Uuid::from_u64_pair(chain_seed, chain_seed.rotate_left(32));
    let ctx = EmitContext {
        run_id,
        narrative_id: sample_narrative_id.clone(),
        maturity: MaturityLevel::Candidate,
        confidence: DEFAULT_SYNTHETIC_CONFIDENCE,
        label_prefix: format!("bistab-nudhy-{k_idx}-"),
        time_anchor: chrono::TimeZone::with_ymd_and_hms(&Utc, 2020, 1, 1, 0, 0, 0).unwrap(),
        step_duration_seconds: 1,
        model: "nudhy".to_string(),
        reuse_entities: Some(needed.iter().copied().collect()),
    };
    let mut sit_rng = ChaCha8Rng::seed_from_u64(chain_seed.wrapping_add(0xC0FFEE));
    for (step, members) in final_state.hyperedges.iter().enumerate() {
        write_synthetic_situation(&ctx, step, members, &mut sit_rng, &target)?;
    }
    let step_offset = final_state.hyperedges.len();
    for (i, members) in fixed_edges.iter().enumerate() {
        write_synthetic_situation(&ctx, step_offset + i, members, &mut sit_rng, &target)?;
    }

    run_bistability_sweep(&target, &sample_narrative_id, params)
}

/// Per-K seed mix — same algebra as Phase 7's `per_sample_seed`.
#[inline]
pub(super) fn per_sample_seed(base_seed: u64, sample_idx: usize) -> u64 {
    base_seed ^ (sample_idx as u64).wrapping_mul(super::fidelity::SAMPLE_SEED_MIX)
}
