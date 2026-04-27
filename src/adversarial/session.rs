//! Wargame session manager — lifecycle, objectives, auto-play.
//!
//! Manages the full wargame lifecycle: create -> configure -> play turns
//! -> evaluate objectives -> archive. Supports auto-play mode where both
//! teams use policy generators.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::synth::hybrid::HybridComponent;

use super::sim_state::*;
use super::types::*;
use super::wargame::*;

// ─── Session ─────────────────────────────────────────────────

/// A wargame session wrapping simulation state + config + history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WargameSession {
    pub session_id: String,
    pub narrative_id: String,
    pub state: SimulationState,
    pub config: WargameConfig,
    pub history: Vec<TurnResult>,
    pub status: SessionStatus,
    pub created_at: DateTime<Utc>,
}

/// Session lifecycle status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    Created,
    Running,
    Completed,
    Archived,
}

/// Civilian-background substrate the wargame runs on top of (EATH Phase 9).
///
/// The wargame's compartmental SMIR model + actor reward dynamics need a
/// population to operate over. By default that's the calling narrative
/// (`None`). Phase 9 adds three opt-in alternatives:
///
/// * [`BackgroundSubstrate::ExistingNarrative`] — fork from a pre-generated
///   synthetic narrative (e.g. produced by an earlier `/synth/generate` job).
///   Use this when the substrate is too large for inline generation
///   ([`SUBSTRATE_INLINE_ENTITY_CAP`] = 500 entities).
/// * [`BackgroundSubstrate::Synthetic`] — calibrate-then-generate inline at
///   session construction. Cap [`SUBSTRATE_INLINE_ENTITY_CAP`].
/// * [`BackgroundSubstrate::SyntheticHybrid`] — mixture-distribution hybrid
///   (Phase 9 flagship). Same inline cap.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum BackgroundSubstrate {
    /// No background substrate — the wargame runs over the existing narrative
    /// directly (current behaviour, equivalent to omitting the field).
    #[default]
    None,
    /// Fork from a pre-existing narrative (real or synthetic). The narrative
    /// id is the substrate's narrative_id; the wargame creates its own
    /// compartments without modifying the source.
    ExistingNarrative { narrative_id: String },
    /// Generate a single-source synthetic substrate inline at session
    /// construction. Caps at [`SUBSTRATE_INLINE_ENTITY_CAP`] entities; for
    /// larger substrates use Phase 4's `POST /synth/generate` job and pass
    /// the result via `ExistingNarrative`.
    Synthetic {
        source_narrative_id: String,
        model: String,
        params: Option<serde_json::Value>,
        seed: Option<u64>,
    },
    /// Generate a hybrid (mixture-distribution) substrate inline at session
    /// construction. Same inline cap as `Synthetic`. The Phase 9 flagship
    /// substrate kind.
    SyntheticHybrid {
        components: Vec<HybridComponent>,
        seed: Option<u64>,
    },
}

/// Inline-generation entity cap for `Synthetic` / `SyntheticHybrid` substrate
/// modes. Phase 9 risks-section rationale: sync session construction means
/// inline generation budgets are tight; bigger substrates must come from a
/// Phase 4 `POST /synth/generate` job (output passed via `ExistingNarrative`).
pub const SUBSTRATE_INLINE_ENTITY_CAP: usize = 500;

/// Wargame configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WargameConfig {
    pub max_turns: usize,
    pub time_step_minutes: u64,
    pub red_objectives: Vec<Objective>,
    pub blue_objectives: Vec<Objective>,
    pub auto_red: bool,
    pub auto_blue: bool,
    /// Optional civilian-background substrate (EATH Phase 9). `None` /
    /// `Some(BackgroundSubstrate::None)` keeps current behaviour (wargame
    /// runs directly over the calling narrative).
    #[serde(default)]
    pub background: Option<BackgroundSubstrate>,
}

impl Default for WargameConfig {
    fn default() -> Self {
        Self {
            max_turns: 20,
            time_step_minutes: 60,
            red_objectives: vec![Objective {
                description: "Maximize R₀ above 2.0".into(),
                metric: ObjectiveMetric::NarrativeR0Above(2.0),
            }],
            blue_objectives: vec![Objective {
                description: "Reduce R₀ below 1.0".into(),
                metric: ObjectiveMetric::NarrativeR0Below(1.0),
            }],
            auto_red: true,
            auto_blue: false,
            background: None,
        }
    }
}

/// A wargame objective with a measurable metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub description: String,
    pub metric: ObjectiveMetric,
}

/// Measurable metric for objective evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveMetric {
    NarrativeR0Above(f64),
    NarrativeR0Below(f64),
    MisinformedAbove(f64),
    MisinformedBelow(f64),
    TurnsCompleted(usize),
}

// ─── Session Lifecycle ───────────────────────────────────────

impl WargameSession {
    /// Create a new wargame session from a hypergraph narrative.
    ///
    /// EATH Phase 9: when `config.background` is `Synthetic` or
    /// `SyntheticHybrid`, the substrate is generated inline at session
    /// construction (capped at [`SUBSTRATE_INLINE_ENTITY_CAP`] entities) and
    /// the wargame forks from the **synthetic** narrative_id rather than
    /// `narrative_id`. The substrate's narrative_id is prefixed with
    /// `wargame-{session_id}-` so it can never collide with the source
    /// narrative_ids and downstream invariant tests can identify it as a
    /// wargame-owned record.
    pub fn create(
        hypergraph: &Hypergraph,
        narrative_id: &str,
        config: WargameConfig,
    ) -> Result<Self> {
        let session_id = Uuid::now_v7().to_string();

        // Resolve the effective substrate narrative id. Defaults to the
        // calling narrative; substrate variants point to a synthetic one.
        let substrate_nid =
            resolve_background_substrate(hypergraph, &session_id, narrative_id, &config.background)?;

        let state = SimulationState::fork_from_hypergraph(
            hypergraph,
            &substrate_nid,
            config.time_step_minutes,
        )?;

        Ok(Self {
            session_id,
            narrative_id: substrate_nid,
            state,
            config,
            history: Vec::new(),
            status: SessionStatus::Created,
            created_at: Utc::now(),
        })
    }

    /// Submit moves for one turn and advance.
    pub fn play_turn(
        &mut self,
        red_moves: &[WargameMove],
        blue_moves: &[WargameMove],
    ) -> Result<TurnResult> {
        if self.status == SessionStatus::Completed || self.status == SessionStatus::Archived {
            return Err(TensaError::Internal("Session is not active".into()));
        }

        self.status = SessionStatus::Running;

        let result = advance_turn(&mut self.state, red_moves, blue_moves)?;

        // Evaluate objectives
        let mut met = Vec::new();
        for obj in self
            .config
            .red_objectives
            .iter()
            .chain(&self.config.blue_objectives)
        {
            if evaluate_objective(&self.state, &obj.metric) {
                met.push(obj.description.clone());
            }
        }

        let mut result = result;
        result.objectives_met = met;
        self.history.push(result.clone());

        // Check if max turns reached
        if self.state.turn_number >= self.config.max_turns {
            self.status = SessionStatus::Completed;
        }

        Ok(result)
    }

    /// Auto-play N turns using policy generators for auto-controlled teams.
    pub fn auto_play(&mut self, num_turns: usize) -> Result<Vec<TurnResult>> {
        let mut results = Vec::with_capacity(num_turns);

        for _ in 0..num_turns {
            if self.status == SessionStatus::Completed {
                break;
            }

            let turn = self.state.turn_number;
            let red_moves = if self.config.auto_red {
                generate_auto_moves(&self.state, Team::Red, turn)
            } else {
                vec![]
            };

            let blue_moves = if self.config.auto_blue {
                generate_auto_moves(&self.state, Team::Blue, turn)
            } else {
                vec![]
            };

            let result = self.play_turn(&red_moves, &blue_moves)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get a summary of the session state.
    pub fn summary(&self) -> SessionSummary {
        let (r0, misinformed, susceptible) = {
            let mut tr0 = 0.0;
            let mut tm = 0.0;
            let mut ts = 0.0;
            let mut c = 0;
            for comp in self.state.compartments.values() {
                tr0 += comp.r0();
                tm += comp.misinformed;
                ts += comp.susceptible;
                c += 1;
            }
            if c > 0 {
                (tr0 / c as f64, tm, ts)
            } else {
                (0.0, 0.0, 0.0)
            }
        };

        let red_objectives_met = self
            .config
            .red_objectives
            .iter()
            .filter(|o| evaluate_objective(&self.state, &o.metric))
            .count();
        let blue_objectives_met = self
            .config
            .blue_objectives
            .iter()
            .filter(|o| evaluate_objective(&self.state, &o.metric))
            .count();

        SessionSummary {
            session_id: self.session_id.clone(),
            status: self.status.clone(),
            turn: self.state.turn_number,
            max_turns: self.config.max_turns,
            r0,
            misinformed,
            susceptible,
            red_objectives_met,
            blue_objectives_met,
            total_moves: self.state.move_log.len(),
        }
    }

    /// Provenance descriptor for the substrate this session is running on
    /// (EATH Phase 9). Retrodiction + reward report renderers consume this
    /// to mark downstream artifacts with substrate kind. Reads only the
    /// `config.background` slot — no recomputation.
    pub fn substrate_provenance(&self) -> SubstrateProvenance {
        match &self.config.background {
            None | Some(BackgroundSubstrate::None) => SubstrateProvenance::Real {
                narrative_id: self.narrative_id.clone(),
            },
            Some(BackgroundSubstrate::ExistingNarrative { narrative_id }) => {
                SubstrateProvenance::Real {
                    narrative_id: narrative_id.clone(),
                }
            }
            Some(BackgroundSubstrate::Synthetic {
                source_narrative_id,
                model,
                seed,
                ..
            }) => SubstrateProvenance::Synthetic {
                source_narrative_id: source_narrative_id.clone(),
                model: model.clone(),
                seed: *seed,
                substrate_narrative_id: self.narrative_id.clone(),
            },
            Some(BackgroundSubstrate::SyntheticHybrid { components, seed }) => {
                SubstrateProvenance::Hybrid {
                    components: components.clone(),
                    seed: *seed,
                    substrate_narrative_id: self.narrative_id.clone(),
                }
            }
        }
    }
}

/// Substrate-provenance descriptor for a wargame session (EATH Phase 9).
///
/// Returned by [`WargameSession::substrate_provenance`]. Retrodiction +
/// reward modules use this to label downstream reports with whether the
/// session ran on real, synthetic, or hybrid data — no logic changes to the
/// retrodiction / reward engines themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SubstrateProvenance {
    /// Session ran directly over an empirical narrative (no synth substrate).
    Real { narrative_id: String },
    /// Session ran over a single-source synthetic substrate.
    Synthetic {
        source_narrative_id: String,
        model: String,
        seed: Option<u64>,
        substrate_narrative_id: String,
    },
    /// Session ran over a hybrid (mixture-distribution) substrate.
    Hybrid {
        components: Vec<HybridComponent>,
        seed: Option<u64>,
        substrate_narrative_id: String,
    },
}

/// Summary of a wargame session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub status: SessionStatus,
    pub turn: usize,
    pub max_turns: usize,
    pub r0: f64,
    pub misinformed: f64,
    pub susceptible: f64,
    pub red_objectives_met: usize,
    pub blue_objectives_met: usize,
    pub total_moves: usize,
}

// ─── Objective Evaluation ────────────────────────────────────

fn evaluate_objective(state: &SimulationState, metric: &ObjectiveMetric) -> bool {
    match metric {
        ObjectiveMetric::NarrativeR0Above(threshold) => {
            state.narratives.values().any(|n| n.r0 > *threshold)
        }
        ObjectiveMetric::NarrativeR0Below(threshold) => {
            state.narratives.values().all(|n| n.r0 < *threshold)
        }
        ObjectiveMetric::MisinformedAbove(threshold) => {
            let total: f64 = state.compartments.values().map(|c| c.misinformed).sum();
            total > *threshold
        }
        ObjectiveMetric::MisinformedBelow(threshold) => {
            let total: f64 = state.compartments.values().map(|c| c.misinformed).sum();
            total < *threshold
        }
        ObjectiveMetric::TurnsCompleted(n) => state.turn_number >= *n,
    }
}

// ─── Auto-Play Move Generation ───────────────────────────────

/// Generate simple heuristic moves for auto-controlled teams.
fn generate_auto_moves(state: &SimulationState, team: Team, turn: usize) -> Vec<WargameMove> {
    let mut moves = Vec::new();

    let actors: Vec<_> = state
        .actors
        .values()
        .filter(|a| a.team == team && a.resources.posts_remaining > 0)
        .collect();

    let platform = state
        .platforms
        .keys()
        .next()
        .cloned()
        .unwrap_or_else(|| "twitter".to_string());

    let narrative = state.narrative_id.clone();

    for actor in actors.iter().take(3) {
        let action = match team {
            Team::Red => WargameAction::PublishContent {
                content: format!("auto-red-turn-{}", turn),
                amplify: turn % 2 == 0,
            },
            Team::Blue => WargameAction::Debunk {
                evidence: format!("auto-blue-turn-{}", turn),
            },
            Team::Grey => WargameAction::WaitAndObserve,
        };

        moves.push(WargameMove {
            turn,
            team,
            actor_id: actor.entity_id,
            action,
            target_narrative: narrative.clone(),
            target_platform: platform.clone(),
        });
    }

    moves
}

// ─── Background Substrate Resolution (EATH Phase 9) ─────────

/// Resolve the substrate narrative the wargame should fork its `SimulationState`
/// from. Default + `None` + `BackgroundSubstrate::None` → the caller's
/// `narrative_id`. Other variants generate or reference an alternate narrative
/// and return its id.
///
/// `Synthetic` / `SyntheticHybrid` paths cap inline generation at
/// [`SUBSTRATE_INLINE_ENTITY_CAP`] entities (Phase 9 risks-section). Errors
/// bubble up before any KV write so a partial failure can't leave half-written
/// substrate records around.
fn resolve_background_substrate(
    hypergraph: &Hypergraph,
    session_id: &str,
    fallback_narrative_id: &str,
    background: &Option<BackgroundSubstrate>,
) -> Result<String> {
    let substrate = match background {
        None | Some(BackgroundSubstrate::None) => return Ok(fallback_narrative_id.to_string()),
        Some(s) => s,
    };

    match substrate {
        BackgroundSubstrate::None => Ok(fallback_narrative_id.to_string()),
        BackgroundSubstrate::ExistingNarrative { narrative_id } => Ok(narrative_id.clone()),
        BackgroundSubstrate::Synthetic {
            source_narrative_id,
            model,
            params,
            seed,
        } => generate_inline_synthetic(
            hypergraph,
            session_id,
            source_narrative_id,
            model,
            params.clone(),
            *seed,
        ),
        BackgroundSubstrate::SyntheticHybrid { components, seed } => {
            generate_inline_hybrid(hypergraph, session_id, components, *seed)
        }
    }
}

fn substrate_output_narrative_id(session_id: &str) -> String {
    format!("wargame-{}-substrate", session_id)
}

fn generate_inline_synthetic(
    hypergraph: &Hypergraph,
    session_id: &str,
    source_narrative_id: &str,
    model: &str,
    params_override: Option<serde_json::Value>,
    seed: Option<u64>,
) -> Result<String> {
    use crate::synth::calibrate::{generate_seed, load_params};
    use crate::synth::eath::generate_with_source;
    use crate::synth::types::SurrogateParams;

    if model != "eath" {
        return Err(TensaError::SynthFailure(format!(
            "wargame inline synthetic substrate: only model='eath' is supported (got '{model}')"
        )));
    }

    // Resolve EathParams: override > calibrated.
    let eath_params = if let Some(raw) = params_override {
        serde_json::from_value::<crate::synth::types::EathParams>(raw).map_err(|e| {
            TensaError::InvalidInput(format!(
                "wargame substrate Synthetic: params override is not a valid EathParams blob: {e}"
            ))
        })?
    } else {
        load_params(hypergraph.store(), source_narrative_id, model)?.ok_or_else(|| {
            TensaError::SynthFailure(format!(
                "wargame substrate Synthetic: no calibrated params for ('{source_narrative_id}', '{model}'); calibrate first"
            ))
        })?
    };

    // Phase 9 risks: inline-generation entity cap.
    if eath_params.a_t_distribution.len() > SUBSTRATE_INLINE_ENTITY_CAP {
        return Err(TensaError::InvalidInput(format!(
            "wargame substrate Synthetic: source has {} entities (> inline cap {}); \
             pre-generate via POST /synth/generate and pass via ExistingNarrative",
            eath_params.a_t_distribution.len(),
            SUBSTRATE_INLINE_ENTITY_CAP
        )));
    }

    let output_nid = substrate_output_narrative_id(session_id);
    let surrogate = SurrogateParams {
        model: model.to_string(),
        params_json: serde_json::to_value(&eath_params).map_err(|e| {
            TensaError::SynthFailure(format!("EathParams → JSON failed: {e}"))
        })?,
        seed: seed.unwrap_or_else(generate_seed),
        // Wargame substrates are short — 100 steps default is sufficient
        // for a SMIR-style civilian-population baseline.
        num_steps: 100,
        label_prefix: format!("wargame-{session_id}-actor"),
    };

    generate_with_source(&surrogate, hypergraph, &output_nid, source_narrative_id)?;
    Ok(output_nid)
}

fn generate_inline_hybrid(
    hypergraph: &Hypergraph,
    session_id: &str,
    components: &[HybridComponent],
    seed: Option<u64>,
) -> Result<String> {
    use crate::synth::calibrate::{generate_seed, load_params};
    use crate::synth::hybrid::generate_hybrid_hypergraph;
    use crate::synth::registry::SurrogateRegistry;

    // Sum source num_entities; cap at the inline budget.
    let mut max_entities = 0_usize;
    for c in components {
        let p = load_params(hypergraph.store(), &c.narrative_id, &c.model)?.ok_or_else(
            || {
                TensaError::SynthFailure(format!(
                    "wargame substrate SyntheticHybrid: no calibrated params for ('{}', '{}'); \
                     calibrate first",
                    c.narrative_id, c.model
                ))
            },
        )?;
        if p.a_t_distribution.len() > max_entities {
            max_entities = p.a_t_distribution.len();
        }
    }
    if max_entities > SUBSTRATE_INLINE_ENTITY_CAP {
        return Err(TensaError::InvalidInput(format!(
            "wargame substrate SyntheticHybrid: max source has {} entities (> inline cap {}); \
             pre-generate via POST /synth/generate-hybrid and pass via ExistingNarrative",
            max_entities, SUBSTRATE_INLINE_ENTITY_CAP
        )));
    }

    let output_nid = substrate_output_narrative_id(session_id);
    let registry = SurrogateRegistry::default();
    generate_hybrid_hypergraph(
        components,
        &output_nid,
        seed.unwrap_or_else(generate_seed),
        100,
        hypergraph,
        &registry,
    )?;
    Ok(output_nid)
}

// ─── KV Storage ──────────────────────────────────────────────

const SESSION_PREFIX: &[u8] = b"adv/wg/";

pub fn store_session(hypergraph: &Hypergraph, session: &WargameSession) -> Result<()> {
    let mut key = SESSION_PREFIX.to_vec();
    key.extend_from_slice(session.session_id.as_bytes());
    let value = serde_json::to_vec(session)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

pub fn load_session(hypergraph: &Hypergraph, session_id: &str) -> Result<Option<WargameSession>> {
    let mut key = SESSION_PREFIX.to_vec();
    key.extend_from_slice(session_id.as_bytes());
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

pub fn list_sessions(hypergraph: &Hypergraph) -> Result<Vec<WargameSession>> {
    let pairs = hypergraph.store().prefix_scan(SESSION_PREFIX)?;
    let mut sessions = Vec::with_capacity(pairs.len());
    for (_, v) in pairs {
        sessions.push(serde_json::from_slice::<WargameSession>(&v)?);
    }
    Ok(sessions)
}

pub fn delete_session(hypergraph: &Hypergraph, session_id: &str) -> Result<()> {
    let mut key = SESSION_PREFIX.to_vec();
    key.extend_from_slice(session_id.as_bytes());
    hypergraph.store().delete(&key)?;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::EntityType;
    use std::sync::Arc;

    fn setup_hg() -> (Hypergraph, Uuid, Uuid) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store);

        let mk = |name: &str| {
            hg.create_entity(crate::types::Entity {
                id: Uuid::now_v7(),
                entity_type: EntityType::Actor,
                properties: serde_json::json!({"name": name}),
                beliefs: None,
                embedding: None,
                maturity: crate::types::MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                provenance: vec![],
                extraction_method: None,
                narrative_id: Some("wg-test".to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            })
            .unwrap()
        };

        let r = mk("Red");
        let b = mk("Blue");
        (hg, r, b)
    }

    fn make_session(hg: &Hypergraph, red_id: Uuid, blue_id: Uuid) -> WargameSession {
        let config = WargameConfig {
            max_turns: 10,
            auto_red: true,
            auto_blue: true,
            ..Default::default()
        };
        let mut session = WargameSession::create(hg, "wg-test", config).unwrap();
        session.state.assign_team(&red_id, Team::Red).unwrap();
        session.state.assign_team(&blue_id, Team::Blue).unwrap();
        session.state.seed_narrative("wg-test", "twitter", 10.0);
        session
    }

    #[test]
    fn test_session_create() {
        let (hg, r, b) = setup_hg();
        let session = make_session(&hg, r, b);
        assert_eq!(session.status, SessionStatus::Created);
        assert_eq!(session.state.turn_number, 0);
    }

    #[test]
    fn test_auto_play_5_turns() {
        let (hg, r, b) = setup_hg();
        let mut session = make_session(&hg, r, b);

        let results = session.auto_play(5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(session.state.turn_number, 5);
        assert_eq!(session.status, SessionStatus::Running);
    }

    #[test]
    fn test_session_completes_at_max_turns() {
        let (hg, r, b) = setup_hg();
        let mut session = make_session(&hg, r, b);

        let results = session.auto_play(20).unwrap();
        assert_eq!(results.len(), 10); // max_turns = 10
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    fn test_cannot_play_after_completed() {
        let (hg, r, b) = setup_hg();
        let mut session = make_session(&hg, r, b);
        session.auto_play(10).unwrap();

        assert_eq!(session.status, SessionStatus::Completed);
        assert!(session.play_turn(&[], &[]).is_err());
    }

    #[test]
    fn test_objective_evaluation() {
        let (hg, r, b) = setup_hg();
        let mut session = make_session(&hg, r, b);

        // Seed heavily so R₀ is high
        session.state.seed_narrative("wg-test", "twitter", 500.0);

        // Play one turn to update R₀
        session.auto_play(1).unwrap();

        let summary = session.summary();
        assert!(summary.r0 > 0.0);
        assert!(summary.total_moves > 0);
    }

    #[test]
    fn test_session_persistence_roundtrip() {
        let (hg, r, b) = setup_hg();
        let mut session = make_session(&hg, r, b);
        session.auto_play(3).unwrap();

        let sid = session.session_id.clone();
        store_session(&hg, &session).unwrap();

        let loaded = load_session(&hg, &sid).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.session_id, sid);
        assert_eq!(loaded.state.turn_number, 3);
        assert_eq!(loaded.history.len(), 3);
    }

    #[test]
    fn test_list_and_delete_sessions() {
        let (hg, r, b) = setup_hg();
        let s1 = make_session(&hg, r, b);
        let s2 = make_session(&hg, r, b);

        store_session(&hg, &s1).unwrap();
        store_session(&hg, &s2).unwrap();

        let all = list_sessions(&hg).unwrap();
        assert_eq!(all.len(), 2);

        delete_session(&hg, &s1.session_id).unwrap();
        let after = list_sessions(&hg).unwrap();
        assert_eq!(after.len(), 1);
    }

    #[test]
    fn test_summary_fields() {
        let (hg, r, b) = setup_hg();
        let session = make_session(&hg, r, b);
        let summary = session.summary();

        assert_eq!(summary.turn, 0);
        assert_eq!(summary.max_turns, 10);
        assert_eq!(summary.total_moves, 0);
    }
}
