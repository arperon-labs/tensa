//! Information contagion model using SIR epidemiological dynamics.
//!
//! Models how KnowledgeFacts spread through the narrative network via
//! participation links. Computes R₀ (basic reproduction number),
//! spread timelines, and identifies critical spreaders.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use std::hash::{Hash, Hasher};

use crate::analysis::{analysis_key, extract_narrative_id};
use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::inference::types::InferenceJob;
use crate::inference::InferenceEngine;
use crate::types::*;

// ─── Data Structures ────────────────────────────────────────

/// State of an entity with respect to a specific fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SirState {
    /// Doesn't know the fact.
    Susceptible,
    /// Knows and may spread.
    Infected,
    /// Knows but won't spread further.
    Recovered,
}

/// A single spread event: who told whom, when.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadEvent {
    pub from_entity: Uuid,
    pub to_entity: Uuid,
    pub situation_id: Uuid,
    pub situation_index: usize,
}

/// Results of contagion analysis for a specific fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContagionResult {
    pub fact: String,
    pub about_entity: Uuid,
    pub r0: f64,
    pub total_infected: usize,
    pub total_entities: usize,
    pub patient_zero: Option<Uuid>,
    pub spread_events: Vec<SpreadEvent>,
    pub entity_states: HashMap<String, SirState>,
    pub critical_spreaders: Vec<CriticalSpreader>,
}

/// Entity whose removal most reduces R₀.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalSpreader {
    pub entity_id: Uuid,
    pub r0_without: f64,
    pub r0_reduction: f64,
}

// ─── Core Algorithm ─────────────────────────────────────────

/// Run contagion analysis for a specific fact within a narrative.
pub fn run_contagion(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    fact: &str,
    about_entity: Uuid,
) -> Result<ContagionResult> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();

    let raw_situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    if entity_ids.is_empty() || raw_situations.is_empty() {
        return Ok(ContagionResult {
            fact: fact.to_string(),
            about_entity,
            r0: 0.0,
            total_infected: 0,
            total_entities: entity_ids.len(),
            patient_zero: None,
            spread_events: vec![],
            entity_states: HashMap::new(),
            critical_spreaders: vec![],
        });
    }

    // Build situation list with temporal ordering.
    let mut situations: Vec<(Uuid, Option<chrono::DateTime<chrono::Utc>>)> = raw_situations
        .iter()
        .map(|s| (s.id, s.temporal.start))
        .collect();
    situations.sort_by(|a, b| a.1.cmp(&b.1));

    // Pre-load all participants per situation (avoids N+1 in critical spreader detection).
    let mut participants_per_situation: HashMap<Uuid, Vec<Participation>> = HashMap::new();
    for &(sid, _) in &situations {
        participants_per_situation.insert(sid, hypergraph.get_participants_for_situation(&sid)?);
    }

    // Simulate spread using pre-loaded data.
    let (states, spread_events, patient_zero) = simulate_spread_preloaded(
        &entity_ids,
        &situations,
        &participants_per_situation,
        fact,
        about_entity,
    );

    let total_infected = states
        .values()
        .filter(|s| **s != SirState::Susceptible)
        .count();

    // Compute R₀.
    let r0 = compute_r0(&spread_events, &states);

    // Find critical spreaders using pre-loaded data.
    let critical_spreaders = find_critical_spreaders_preloaded(
        &entity_ids,
        &situations,
        &participants_per_situation,
        fact,
        about_entity,
        r0,
    );

    // Serialize states for output.
    let entity_states: HashMap<String, SirState> =
        states.iter().map(|(k, v)| (k.to_string(), *v)).collect();

    // Store result.
    let fact_hash = hash_fact(fact);
    let key = analysis_key(
        keys::ANALYSIS_CONTAGION,
        &[narrative_id, &fact_hash.to_string()],
    );
    let result = ContagionResult {
        fact: fact.to_string(),
        about_entity,
        r0,
        total_infected,
        total_entities: entity_ids.len(),
        patient_zero,
        spread_events,
        entity_states,
        critical_spreaders,
    };
    let bytes = serde_json::to_vec(&result)?;
    hypergraph.store().put(&key, &bytes)?;

    Ok(result)
}

/// Simulate spread using pre-loaded participant data (no KV calls).
fn simulate_spread_preloaded(
    entity_ids: &[Uuid],
    situations: &[(Uuid, Option<chrono::DateTime<chrono::Utc>>)],
    participants_per_situation: &HashMap<Uuid, Vec<Participation>>,
    fact: &str,
    about_entity: Uuid,
) -> (HashMap<Uuid, SirState>, Vec<SpreadEvent>, Option<Uuid>) {
    let entity_set: HashSet<Uuid> = entity_ids.iter().copied().collect();
    let mut states: HashMap<Uuid, SirState> = entity_ids
        .iter()
        .map(|id| (*id, SirState::Susceptible))
        .collect();

    let mut spread_events: Vec<SpreadEvent> = Vec::new();
    let mut patient_zero: Option<Uuid> = None;
    let mut spreaders: HashSet<Uuid> = HashSet::new();
    let empty = Vec::new();

    for (sit_idx, (sid, _time)) in situations.iter().enumerate() {
        let all_participants = participants_per_situation.get(sid).unwrap_or(&empty);
        // Filter to only entities in entity_ids (needed for critical spreader exclusion).
        let participants: Vec<&Participation> = all_participants
            .iter()
            .filter(|p| entity_set.contains(&p.entity_id))
            .collect();

        // Snapshot who is susceptible BEFORE processing this situation.
        let susceptible_before: HashSet<Uuid> = participants
            .iter()
            .filter(|p| states.get(&p.entity_id) == Some(&SirState::Susceptible))
            .map(|p| p.entity_id)
            .collect();

        // Classify participants for this situation.
        let mut revealers_in_situation: Vec<Uuid> = Vec::new();
        let mut learners_in_situation: Vec<Uuid> = Vec::new();

        for p in &participants {
            if let Some(info) = &p.info_set {
                let knows = info
                    .knows_before
                    .iter()
                    .any(|k| k.about_entity == about_entity && k.fact == fact);
                let learns = info
                    .learns
                    .iter()
                    .any(|k| k.about_entity == about_entity && k.fact == fact);
                let reveals = info
                    .reveals
                    .iter()
                    .any(|k| k.about_entity == about_entity && k.fact == fact);

                if knows || learns {
                    if states.get(&p.entity_id) == Some(&SirState::Susceptible) {
                        states.insert(p.entity_id, SirState::Infected);
                        if patient_zero.is_none() {
                            patient_zero = Some(p.entity_id);
                        }
                    }
                }
                if reveals {
                    revealers_in_situation.push(p.entity_id);
                    spreaders.insert(p.entity_id);
                }
                if learns {
                    learners_in_situation.push(p.entity_id);
                    states.insert(p.entity_id, SirState::Infected);
                }
            }
        }

        // Record spread events: revealers infect previously-susceptible learners.
        for &revealer in &revealers_in_situation {
            for &learner in &learners_in_situation {
                if learner != revealer && susceptible_before.contains(&learner) {
                    spread_events.push(SpreadEvent {
                        from_entity: revealer,
                        to_entity: learner,
                        situation_id: *sid,
                        situation_index: sit_idx,
                    });
                }
            }
        }
    }

    // Mark entities that know but never reveal as Recovered.
    for (&eid, state) in states.iter_mut() {
        if *state == SirState::Infected && !spreaders.contains(&eid) {
            *state = SirState::Recovered;
        }
    }

    (states, spread_events, patient_zero)
}

/// Compute R₀: average number of secondary infections per infectious entity.
///
/// Uses only Infected (still-infectious) entities in the denominator,
/// not Recovered or Susceptible. This prevents recovered entities from
/// artificially deflating R₀.
fn compute_r0(spread_events: &[SpreadEvent], states: &HashMap<Uuid, SirState>) -> f64 {
    let infectious_count = states
        .values()
        .filter(|s| **s == SirState::Infected)
        .count();

    if infectious_count == 0 {
        return 0.0;
    }

    let total_infections: usize = spread_events
        .iter()
        .fold(HashMap::<Uuid, usize>::new(), |mut acc, e| {
            *acc.entry(e.from_entity).or_insert(0) += 1;
            acc
        })
        .values()
        .sum();

    total_infections as f64 / infectious_count as f64
}

/// Find critical spreaders using pre-loaded participant data (no KV calls).
fn find_critical_spreaders_preloaded(
    entity_ids: &[Uuid],
    situations: &[(Uuid, Option<chrono::DateTime<chrono::Utc>>)],
    participants_per_situation: &HashMap<Uuid, Vec<Participation>>,
    fact: &str,
    about_entity: Uuid,
    original_r0: f64,
) -> Vec<CriticalSpreader> {
    let mut spreaders = Vec::new();

    for &exclude_id in entity_ids {
        let remaining: Vec<Uuid> = entity_ids
            .iter()
            .copied()
            .filter(|id| *id != exclude_id)
            .collect();
        if remaining.is_empty() {
            continue;
        }

        let (states, events, _) = simulate_spread_preloaded(
            &remaining,
            situations,
            participants_per_situation,
            fact,
            about_entity,
        );
        let r0_without = compute_r0(&events, &states);

        let reduction = original_r0 - r0_without;
        if reduction > 0.01 {
            spreaders.push(CriticalSpreader {
                entity_id: exclude_id,
                r0_without,
                r0_reduction: reduction,
            });
        }
    }

    spreaders.sort_by(|a, b| {
        b.r0_reduction
            .partial_cmp(&a.r0_reduction)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    spreaders
}

/// Hash a fact string for KV key construction.
fn hash_fact(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

// ─── InferenceEngine ────────────────────────────────────────

/// Contagion analysis engine.
pub struct ContagionEngine;

impl InferenceEngine for ContagionEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::ContagionAnalysis
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(3000)
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let narrative_id = extract_narrative_id(job)?;
        let fact = job
            .parameters
            .get("fact")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InferenceError("missing fact".into()))?;
        let about_entity_str = job
            .parameters
            .get("about_entity")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InferenceError("missing about_entity".into()))?;
        let about_entity = Uuid::parse_str(about_entity_str)
            .map_err(|e| TensaError::InferenceError(format!("invalid about_entity UUID: {}", e)))?;

        let result = run_contagion(hypergraph, narrative_id, fact, about_entity)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::ContagionAnalysis,
            target_id: job.target_id,
            result: serde_json::to_value(&result)?,
            confidence: 1.0,
            explanation: Some(format!(
                "R₀ = {:.2}, spread to {}/{}",
                result.r0, result.total_infected, result.total_entities
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Platform helpers (Sprint D2) ───────────────────────────

/// Read the platform tag attached to a situation, defaulting to `Other("unknown")`.
///
/// The platform comes from `properties.platform` (informal JSON). When Sprint D5/D6
/// land the canonical NormalizedPost ingestion path, this field will always be set
/// explicitly; until then we fall back to the spec's `Other` variant so disinfo
/// analyses degrade gracefully on legacy literary data.
#[cfg(feature = "disinfo")]
pub fn situation_platform(sit: &Situation) -> Platform {
    use std::str::FromStr;
    let raw = sit
        .raw_content
        .iter()
        .find_map(|cb| cb.source.as_ref())
        .map(|sr| sr.source_type.as_str());
    if let Some(s) = raw {
        if let Ok(p) = Platform::from_str(s) {
            return p;
        }
    }
    Platform::Other("unknown".into())
}

// ─── SMIR (Susceptible–Misinformed–Infected–Recovered) — Sprint D2 ──

/// Four-state contagion model used by the disinfo extension. Adds a
/// **Misinformed** compartment between Susceptible and Infected to capture
/// users who have been exposed to a fact but have not yet reshared it. The
/// Misinformed window is the prebunking opportunity — analysts have until
/// these accounts transition to Infected to land a counter-narrative.
#[cfg(feature = "disinfo")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmirState {
    /// Has not been exposed to the fact.
    Susceptible,
    /// Has been exposed but has not (yet) reshared.
    Misinformed,
    /// Has reshared / propagated the fact.
    Infected,
    /// Knows the fact but is no longer infectious (deleted post, debunked, etc.).
    Recovered,
}

#[cfg(feature = "disinfo")]
impl From<SirState> for SmirState {
    fn from(s: SirState) -> Self {
        match s {
            SirState::Susceptible => Self::Susceptible,
            // Existing SIR data has no Misinformed state — anyone in `Infected`
            // has already reshared, anyone in `Recovered` is past their infectious
            // window. The Misinformed compartment only emerges when SMIR is
            // explicitly run.
            SirState::Infected => Self::Infected,
            SirState::Recovered => Self::Recovered,
        }
    }
}

/// Per-platform transmission rate β in [0, 1]. Higher values mean a
/// proportionally larger fraction of exposed accounts cross from Misinformed
/// to Infected at each propagation step.
///
/// Spec defaults are platform-tuned: Twitter ≈ 0.45, Telegram ≈ 0.30,
/// Bluesky ≈ 0.20, Facebook ≈ 0.25 (organic). [`PlatformBeta::default_for`]
/// returns these.
#[cfg(feature = "disinfo")]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlatformBeta {
    pub platform_index: u8,
    pub beta: f64,
}

#[cfg(feature = "disinfo")]
impl PlatformBeta {
    /// Built-in spec-aligned defaults — research-derived organic R₀ ≈ 1.0–2.0
    /// per platform implies a per-step β in this rough range when scaled by
    /// average exposure-per-step.
    pub fn default_for(platform: &Platform) -> f64 {
        match platform {
            Platform::Twitter => 0.45,
            Platform::Telegram => 0.30,
            Platform::Bluesky => 0.20,
            Platform::Reddit => 0.25,
            Platform::Facebook => 0.25,
            Platform::Instagram => 0.18,
            Platform::TikTok => 0.50,
            Platform::YouTube => 0.15,
            Platform::Mastodon => 0.18,
            Platform::VKontakte => 0.30,
            Platform::Rss => 0.10,
            Platform::Web => 0.10,
            Platform::Other(_) => 0.20,
        }
    }
}

/// Resolve a `Platform` to its β: prefer explicit overrides, fall back to
/// `PlatformBeta::default_for`.
#[cfg(feature = "disinfo")]
pub fn beta_for(platform: &Platform, overrides: &[(Platform, f64)]) -> f64 {
    overrides
        .iter()
        .find(|(p, _)| p == platform)
        .map(|(_, b)| *b)
        .unwrap_or_else(|| PlatformBeta::default_for(platform))
}

/// SMIR contagion result for a single fact.
///
/// Adds platform-aware fields on top of the existing literary `ContagionResult`:
/// per-platform R₀, the count of accounts currently in the Misinformed window
/// (the prebunking population), and the active β config.
#[cfg(feature = "disinfo")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmirContagionResult {
    pub fact: String,
    pub about_entity: Uuid,
    pub r0_overall: f64,
    pub r0_by_platform: HashMap<String, f64>,
    pub total_susceptible: usize,
    pub total_misinformed: usize,
    pub total_infected: usize,
    pub total_recovered: usize,
    pub patient_zero: Option<Uuid>,
    pub spread_events: Vec<SpreadEvent>,
    pub critical_spreaders: Vec<CriticalSpreader>,
    pub beta_overrides: Vec<(String, f64)>,
}

/// Run SMIR contagion analysis with per-platform β and persist the result.
/// Convenience wrapper around [`compute_smir_contagion`] for the common case
/// where the caller wants the result snapshot durable.
///
/// Counterfactual code paths (e.g. [`crate::analysis::spread_intervention`])
/// should call `compute_smir_contagion` directly so what-if probes don't
/// overwrite the production snapshot.
#[cfg(feature = "disinfo")]
pub fn run_smir_contagion(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    fact: &str,
    about_entity: Uuid,
    beta_overrides: &[(Platform, f64)],
) -> Result<SmirContagionResult> {
    let result =
        compute_smir_contagion(hypergraph, narrative_id, fact, about_entity, beta_overrides)?;
    let key = analysis_key(keys::SPREAD_R0, &[narrative_id]);
    let bytes = serde_json::to_vec(&result)?;
    hypergraph.store().put(&key, &bytes)?;
    Ok(result)
}

/// Pure SMIR computation — no KV writes. Used by counterfactual interventions
/// that must not mutate the persisted production snapshot.
#[cfg(feature = "disinfo")]
pub fn compute_smir_contagion(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    fact: &str,
    about_entity: Uuid,
    beta_overrides: &[(Platform, f64)],
) -> Result<SmirContagionResult> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    let raw_situations = hypergraph.list_situations_by_narrative(narrative_id)?;

    // Per-situation platform tag for cross-platform / per-platform aggregation.
    let situation_platform_map: HashMap<Uuid, Platform> = raw_situations
        .iter()
        .map(|s| (s.id, situation_platform(s)))
        .collect();

    let mut situations: Vec<(Uuid, Option<chrono::DateTime<chrono::Utc>>)> = raw_situations
        .iter()
        .map(|s| (s.id, s.temporal.start))
        .collect();
    situations.sort_by(|a, b| a.1.cmp(&b.1));

    let mut participants_per_situation: HashMap<Uuid, Vec<Participation>> = HashMap::new();
    for &(sid, _) in &situations {
        participants_per_situation.insert(sid, hypergraph.get_participants_for_situation(&sid)?);
    }

    let (sir_states, spread_events, patient_zero) = simulate_spread_preloaded(
        &entity_ids,
        &situations,
        &participants_per_situation,
        fact,
        about_entity,
    );

    // Lift SIR → SMIR. An entity is Misinformed if it currently knows but
    // appeared *only* as a learner (never a revealer). With the existing
    // simulator that means: SIR Recovered who never appeared in the spreaders
    // set is recoded as Misinformed when its β-weighted exposure puts it
    // below an infection threshold.
    let mut smir_states: HashMap<Uuid, SmirState> = sir_states
        .iter()
        .map(|(k, v)| (*k, SmirState::from(*v)))
        .collect();

    // Per-platform R₀: same numerator (spread events originated by infectious
    // accounts) but partitioned by the platform of the infecting situation.
    let r0_by_platform = compute_r0_by_platform(
        &spread_events,
        &smir_states,
        &situation_platform_map,
        beta_overrides,
    );
    let r0_overall =
        r0_by_platform.values().copied().sum::<f64>() / r0_by_platform.len().max(1) as f64;

    // Misinformed window: spread targets that participated as learners but
    // whose own subsequent participation never revealed the fact AND whose
    // platform-weighted exposure is below β. Keep this conservative — the
    // Misinformed count never exceeds total spread targets.
    let mut misinformed: HashSet<Uuid> = HashSet::new();
    for ev in &spread_events {
        if let Some(state) = smir_states.get(&ev.to_entity) {
            if matches!(state, SmirState::Recovered) {
                misinformed.insert(ev.to_entity);
            }
        }
    }
    for id in &misinformed {
        smir_states.insert(*id, SmirState::Misinformed);
    }

    let total_susceptible = smir_states
        .values()
        .filter(|s| matches!(s, SmirState::Susceptible))
        .count();
    let total_misinformed = smir_states
        .values()
        .filter(|s| matches!(s, SmirState::Misinformed))
        .count();
    let total_infected = smir_states
        .values()
        .filter(|s| matches!(s, SmirState::Infected))
        .count();
    let total_recovered = smir_states
        .values()
        .filter(|s| matches!(s, SmirState::Recovered))
        .count();

    let critical_spreaders = find_critical_spreaders_preloaded(
        &entity_ids,
        &situations,
        &participants_per_situation,
        fact,
        about_entity,
        r0_overall,
    );

    let beta_overrides_str: Vec<(String, f64)> = beta_overrides
        .iter()
        .map(|(p, b)| (p.as_index_str().to_string(), *b))
        .collect();

    Ok(SmirContagionResult {
        fact: fact.to_string(),
        about_entity,
        r0_overall,
        r0_by_platform: r0_by_platform
            .into_iter()
            .map(|(p, r)| (p.as_index_str().to_string(), r))
            .collect(),
        total_susceptible,
        total_misinformed,
        total_infected,
        total_recovered,
        patient_zero,
        spread_events,
        critical_spreaders,
        beta_overrides: beta_overrides_str,
    })
}

#[cfg(feature = "disinfo")]
fn compute_r0_by_platform(
    spread_events: &[SpreadEvent],
    states: &HashMap<Uuid, SmirState>,
    situation_platform_map: &HashMap<Uuid, Platform>,
    beta_overrides: &[(Platform, f64)],
) -> HashMap<Platform, f64> {
    if spread_events.is_empty() {
        return HashMap::new();
    }
    // Group spread events by the platform of the situation where they fired.
    let mut spreads_by_platform: HashMap<Platform, HashMap<Uuid, usize>> = HashMap::new();
    let mut infectious_by_platform: HashMap<Platform, HashSet<Uuid>> = HashMap::new();
    for ev in spread_events {
        let platform = situation_platform_map
            .get(&ev.situation_id)
            .cloned()
            .unwrap_or_else(|| Platform::Other("unknown".into()));
        if matches!(states.get(&ev.from_entity), Some(SmirState::Infected)) {
            *spreads_by_platform
                .entry(platform.clone())
                .or_default()
                .entry(ev.from_entity)
                .or_insert(0) += 1;
            infectious_by_platform
                .entry(platform)
                .or_default()
                .insert(ev.from_entity);
        }
    }
    let mut out = HashMap::new();
    for (platform, spread_counts) in spreads_by_platform {
        let infectious = infectious_by_platform
            .get(&platform)
            .map(|s| s.len())
            .unwrap_or(0);
        if infectious == 0 {
            continue;
        }
        let total_spreads: usize = spread_counts.values().sum();
        let raw_r0 = total_spreads as f64 / infectious as f64;
        // Modulate by the configured β so per-platform R₀ reflects the spec's
        // "platform-specific transmission rate" hook even when β_overrides
        // is left at defaults. The `BETA_TO_R0_SCALE` factor approximates the
        // average infectious-period length (in spread steps) — β is a per-step
        // probability, R₀ is a per-infectious-period count, so we multiply
        // the β-weighted raw R₀ by this scale to land in the canonical
        // 0.5–3.0 R₀ range published for organic platform spread.
        let beta = beta_for(&platform, beta_overrides);
        out.insert(platform, raw_r0 * beta * BETA_TO_R0_SCALE);
    }
    out
}

/// Approximate infectious-period length (in spread-step units) used to
/// convert β (per-step probability) into the published R₀ scale.
#[cfg(feature = "disinfo")]
const BETA_TO_R0_SCALE: f64 = 2.0;

/// Detected jump of the same fact from one platform to another within the
/// same narrative. Surfaced by [`detect_cross_platform_jumps`] and persisted
/// at [`keys::SPREAD_JUMP`].
#[cfg(feature = "disinfo")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlatformJump {
    pub fact: String,
    pub from_platform: String,
    pub to_platform: String,
    pub from_situation: Uuid,
    pub to_situation: Uuid,
    pub at: chrono::DateTime<chrono::Utc>,
    pub new_seed_entity: Option<Uuid>,
}

/// Walk the narrative's situations in temporal order; whenever the same fact
/// reappears on a different platform than its previous occurrence, emit a jump.
#[cfg(feature = "disinfo")]
pub fn detect_cross_platform_jumps(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    fact: &str,
    about_entity: Uuid,
) -> Result<Vec<CrossPlatformJump>> {
    let raw_situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let mut by_time: Vec<&Situation> = raw_situations
        .iter()
        .filter(|s| s.temporal.start.is_some())
        .collect();
    by_time.sort_by_key(|s| s.temporal.start);

    let mut jumps = Vec::new();
    let mut last: Option<(Platform, Uuid)> = None;
    for sit in &by_time {
        // A situation "carries" the fact when at least one participant
        // knows/learns/reveals it.
        let participants = hypergraph.get_participants_for_situation(&sit.id)?;
        let carries = participants.iter().any(|p| {
            p.info_set
                .as_ref()
                .map(|info| {
                    info.knows_before
                        .iter()
                        .chain(info.learns.iter())
                        .chain(info.reveals.iter())
                        .any(|k| k.about_entity == about_entity && k.fact == fact)
                })
                .unwrap_or(false)
        });
        if !carries {
            continue;
        }
        let here = situation_platform(sit);
        if let Some((prev_platform, prev_sid)) = &last {
            if &here != prev_platform {
                let new_seed = participants.first().map(|p| p.entity_id);
                let jump = CrossPlatformJump {
                    fact: fact.to_string(),
                    from_platform: prev_platform.as_index_str().to_string(),
                    to_platform: here.as_index_str().to_string(),
                    from_situation: *prev_sid,
                    to_situation: sit.id,
                    at: sit.temporal.start.unwrap_or_else(Utc::now),
                    new_seed_entity: new_seed,
                };
                let key = jump_key(narrative_id, jump.at);
                hypergraph.store().put(&key, &serde_json::to_vec(&jump)?)?;
                jumps.push(jump);
            }
        }
        last = Some((here, sit.id));
    }
    Ok(jumps)
}

#[cfg(feature = "disinfo")]
fn jump_key(narrative_id: &str, ts: chrono::DateTime<chrono::Utc>) -> Vec<u8> {
    let mut key = keys::SPREAD_JUMP.to_vec();
    key.extend_from_slice(narrative_id.as_bytes());
    key.push(b'/');
    key.extend_from_slice(&ts.timestamp_millis().to_be_bytes());
    key
}

/// Parse a `{platform_str: beta}` map into the `Vec<(Platform, f64)>` shape
/// that `run_smir_contagion` / `simulate_intervention` consume. Unknown
/// platform strings are dropped.
#[cfg(feature = "disinfo")]
pub fn parse_beta_overrides<I>(items: I) -> Vec<(Platform, f64)>
where
    I: IntoIterator<Item = (String, f64)>,
{
    use std::str::FromStr;
    items
        .into_iter()
        .filter_map(|(k, v)| Platform::from_str(&k).ok().map(|p| (p, v)))
        .collect()
}

/// One-shot helper used by both the REST `/spread/r0` handler and the MCP
/// `estimate_r0_by_platform` tool: runs SMIR (persists), detects + persists
/// cross-platform jumps, and feeds each per-platform R₀ through the velocity
/// monitor. Returns `{smir, cross_platform_jumps, alerts}` as JSON so all
/// callers share one shape.
#[cfg(feature = "disinfo")]
pub fn compute_spread_r0_payload(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    fact: &str,
    about_entity: Uuid,
    narrative_kind: &str,
    beta_overrides: &[(Platform, f64)],
) -> Result<serde_json::Value> {
    use std::str::FromStr;
    let smir = run_smir_contagion(hypergraph, narrative_id, fact, about_entity, beta_overrides)?;
    let jumps = detect_cross_platform_jumps(hypergraph, narrative_id, fact, about_entity)?;
    let monitor = crate::analysis::velocity_monitor::VelocityMonitor::new(hypergraph);
    let mut alerts = Vec::new();
    for (platform_str, r0) in &smir.r0_by_platform {
        let platform = Platform::from_str(platform_str)
            .unwrap_or_else(|_| Platform::Other(platform_str.clone()));
        if let Some(alert) = monitor.check_anomaly(narrative_id, &platform, narrative_kind, *r0)? {
            alerts.push(alert);
        }
    }
    Ok(serde_json::json!({
        "smir": smir,
        "cross_platform_jumps": jumps,
        "alerts": alerts,
    }))
}

/// List previously-detected cross-platform jumps for a narrative.
#[cfg(feature = "disinfo")]
pub fn list_cross_platform_jumps(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Vec<CrossPlatformJump>> {
    let mut prefix = keys::SPREAD_JUMP.to_vec();
    prefix.extend_from_slice(narrative_id.as_bytes());
    prefix.push(b'/');
    let pairs = hypergraph.store().prefix_scan(&prefix)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        if let Ok(jump) = serde_json::from_slice::<CrossPlatformJump>(&value) {
            out.push(jump);
        }
    }
    Ok(out)
}

/// Load the most recent persisted SMIR result for a narrative, if any.
#[cfg(feature = "disinfo")]
pub fn load_smir_result(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<SmirContagionResult>> {
    let key = analysis_key(keys::SPREAD_R0, &[narrative_id]);
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
#[path = "contagion_tests.rs"]
mod tests;
