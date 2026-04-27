//! Simulation State — mutable snapshot of the information environment.
//!
//! The hypergraph IS the analytical store; `SimulationState` is a mutable
//! fork that evolves per wargame turn without corrupting the analytical data.
//! Supports snapshotting and branching for counterfactual exploration.

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::types::{EntityType, Role};

use super::types::*;

// ─── Core State ──────────────────────────────────────────────

/// A mutable snapshot of the information environment at time T.
/// Forked from the analytical hypergraph but independently mutable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    /// Unique simulation identifier.
    pub id: SimulationId,
    /// Source narrative this simulation was forked from.
    pub narrative_id: String,
    /// Current turn number (0-indexed).
    pub turn_number: usize,
    /// Current simulated time.
    pub current_time: DateTime<Utc>,
    /// Time step per turn.
    pub time_step_minutes: u64,

    /// Actors in the simulation, keyed by entity UUID.
    pub actors: HashMap<Uuid, SimActor>,
    /// Narrative threads being tracked.
    pub narratives: HashMap<String, SimNarrative>,
    /// Platform states.
    pub platforms: HashMap<String, PlatformState>,
    /// Audience model.
    pub audience: AudienceModel,

    /// SMIR compartments per "narrative|platform" key.
    pub compartments: HashMap<String, SmirCompartments>,

    /// Move history.
    pub move_log: Vec<MoveLogEntry>,
    /// Saved state snapshots for counterfactual branching.
    /// Excluded from serialization to prevent quadratic growth
    /// (each snapshot would otherwise contain all previous snapshots).
    #[serde(skip)]
    pub snapshots: Vec<StateSnapshot>,
}

/// An actor in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimActor {
    pub entity_id: Uuid,
    pub name: String,
    pub team: Team,
    pub archetype: Option<String>,
    pub resources: ActorResources,
    /// Number of actions taken this turn.
    pub actions_this_turn: usize,
}

/// Resources available to an actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorResources {
    /// Posts remaining today.
    pub posts_remaining: usize,
    /// Active accounts on each platform.
    pub active_accounts: HashMap<String, usize>,
    /// Budget units remaining.
    pub budget_remaining: f64,
}

impl Default for ActorResources {
    fn default() -> Self {
        Self {
            posts_remaining: 50,
            active_accounts: HashMap::new(),
            budget_remaining: 100.0,
        }
    }
}

/// A narrative thread in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimNarrative {
    pub narrative_id: String,
    pub description: String,
    /// Current aggregate R₀ across all platforms.
    pub r0: f64,
    /// Total reach (number of people exposed).
    pub total_reach: usize,
}

/// SMIR compartments for a (narrative, platform) pair.
///
/// S = Susceptible, M = Misinformed, I = Inoculated, R = Recovered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmirCompartments {
    pub susceptible: f64,
    pub misinformed: f64,
    pub inoculated: f64,
    pub recovered: f64,
    /// Transmission rate (β) for this platform.
    pub beta: f64,
    /// Recovery rate (γ) — rate at which misinformed encounter corrections.
    pub gamma: f64,
    /// Inoculation rate (δ) — rate of prebunking effectiveness.
    pub delta: f64,
}

impl SmirCompartments {
    /// Create initial compartments for a population.
    pub fn new(population: f64, beta: f64) -> Self {
        Self {
            susceptible: population,
            misinformed: 0.0,
            inoculated: 0.0,
            recovered: 0.0,
            beta,
            gamma: 0.05,
            delta: 0.01,
        }
    }

    /// Total population (should remain constant).
    pub fn total(&self) -> f64 {
        self.susceptible + self.misinformed + self.inoculated + self.recovered
    }

    /// Compute R₀ = β × S / (γ × N).
    pub fn r0(&self) -> f64 {
        let n = self.total();
        if n <= 0.0 || self.gamma <= 0.0 {
            return 0.0;
        }
        self.beta * self.susceptible / (self.gamma * n)
    }

    /// Advance one time step using Euler's method.
    ///
    /// dS/dt = -β·S·M/N - δ·S
    /// dM/dt = +β·S·M/N - γ·M
    /// dI/dt = +δ·S
    /// dR/dt = +γ·M
    pub fn step(&mut self, dt: f64) {
        let n = self.total();
        if n <= 0.0 {
            return;
        }

        let infection = self.beta * self.susceptible * self.misinformed / n * dt;
        let recovery = self.gamma * self.misinformed * dt;
        let inoculation = self.delta * self.susceptible * dt;

        // Jointly cap outflows from susceptible to prevent going negative
        let total_out = infection + inoculation;
        let (infection, inoculation) = if total_out > self.susceptible && total_out > 0.0 {
            let scale = self.susceptible / total_out;
            (infection * scale, inoculation * scale)
        } else {
            (infection, inoculation)
        };
        let recovery = recovery.min(self.misinformed);

        self.susceptible -= infection + inoculation;
        self.misinformed += infection - recovery;
        self.inoculated += inoculation;
        self.recovered += recovery;

        // Clamp to non-negative
        self.susceptible = self.susceptible.max(0.0);
        self.misinformed = self.misinformed.max(0.0);
        self.inoculated = self.inoculated.max(0.0);
        self.recovered = self.recovered.max(0.0);
    }

    /// Seed initial misinformed population.
    pub fn seed_misinformed(&mut self, count: f64) {
        let transfer = count.min(self.susceptible);
        self.susceptible -= transfer;
        self.misinformed += transfer;
    }
}

/// Audience model for the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudienceModel {
    pub total_population: usize,
    pub segments: Vec<AudienceSegment>,
}

/// An audience segment with susceptibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudienceSegment {
    pub name: String,
    pub size: usize,
    /// Susceptibility to misinformation (0.0-1.0).
    pub susceptibility: f64,
    /// Current belief state per narrative (narrative_id -> belief strength).
    pub beliefs: HashMap<String, f64>,
}

/// A logged move in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveLogEntry {
    pub turn: usize,
    pub team: Team,
    pub actor_id: Uuid,
    pub action_type: ActionType,
    pub platform: String,
    pub narrative_id: String,
    pub timestamp: DateTime<Utc>,
}

/// A saved snapshot for counterfactual branching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub snapshot_id: String,
    pub turn_number: usize,
    pub timestamp: DateTime<Utc>,
    /// Serialized state at this point.
    pub state_json: serde_json::Value,
}

// ─── SimulationState Methods ─────────────────────────────────

impl SimulationState {
    /// Fork a new simulation from the analytical hypergraph.
    ///
    /// Extracts actors, narratives, and creates default platform/audience state.
    pub fn fork_from_hypergraph(
        hypergraph: &Hypergraph,
        narrative_id: &str,
        time_step_minutes: u64,
    ) -> Result<Self> {
        let sim_id = Uuid::now_v7().to_string();

        // Use narrative index when available; wildcard falls back to full scan
        let entities = if narrative_id == "*" {
            hypergraph.list_entities_by_type(&EntityType::Actor)?
        } else {
            hypergraph
                .list_entities_by_narrative(narrative_id)?
                .into_iter()
                .filter(|e| e.entity_type == EntityType::Actor)
                .collect()
        };
        let mut actors = HashMap::new();

        for entity in &entities {
            let name = entity
                .properties
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string();

            actors.insert(
                entity.id,
                SimActor {
                    entity_id: entity.id,
                    name,
                    team: Team::Grey,
                    archetype: None,
                    resources: ActorResources::default(),
                    actions_this_turn: 0,
                },
            );
        }

        // Create default narrative entry
        let mut narratives = HashMap::new();
        narratives.insert(
            narrative_id.to_string(),
            SimNarrative {
                narrative_id: narrative_id.to_string(),
                description: String::new(),
                r0: 0.0,
                total_reach: 0,
            },
        );

        // Default platform states with literature-calibrated betas
        let platforms = default_platforms();

        // Default audience
        let audience = AudienceModel {
            total_population: 10_000,
            segments: vec![AudienceSegment {
                name: "general".to_string(),
                size: 10_000,
                susceptibility: 0.5,
                beliefs: HashMap::new(),
            }],
        };

        // Initialize SMIR compartments for each (narrative, platform)
        let mut compartments = HashMap::new();
        for (platform_name, ps) in &platforms {
            let pop = audience.total_population as f64 / platforms.len() as f64;
            compartments.insert(
                compartment_key(narrative_id, platform_name),
                SmirCompartments::new(pop, ps.beta),
            );
        }

        Ok(Self {
            id: sim_id,
            narrative_id: narrative_id.to_string(),
            turn_number: 0,
            current_time: Utc::now(),
            time_step_minutes,
            actors,
            narratives,
            platforms,
            audience,
            compartments,
            move_log: Vec::new(),
            snapshots: Vec::new(),
        })
    }

    /// Advance SMIR compartments by one time step.
    pub fn evolve_smir(&mut self) {
        let dt = self.time_step_minutes as f64 / 60.0; // Convert to hours
        for comp in self.compartments.values_mut() {
            comp.step(dt);
        }
        // Update aggregate R₀ per narrative
        for (narrative_id, nar) in &mut self.narratives {
            let mut total_r0 = 0.0;
            let mut count = 0;
            let prefix = format!("{}|", narrative_id);
            for (key, comp) in &self.compartments {
                if key.starts_with(&prefix) {
                    total_r0 += comp.r0();
                    count += 1;
                }
            }
            nar.r0 = if count > 0 {
                total_r0 / count as f64
            } else {
                0.0
            };
        }
    }

    /// Save a snapshot of the current state for counterfactual branching.
    pub fn snapshot(&mut self) -> Result<String> {
        let snapshot_id = Uuid::now_v7().to_string();
        let turn = self.turn_number;
        let ts = self.current_time;
        let state_json = serde_json::to_value(&*self)?;
        self.snapshots.push(StateSnapshot {
            snapshot_id: snapshot_id.clone(),
            turn_number: turn,
            timestamp: ts,
            state_json,
        });
        Ok(snapshot_id)
    }

    /// Branch from a saved snapshot, creating an independent copy.
    pub fn branch(snapshot_id: &str, snapshots: &[StateSnapshot]) -> Result<Self> {
        let snap = snapshots
            .iter()
            .find(|s| s.snapshot_id == snapshot_id)
            .ok_or_else(|| TensaError::Internal(format!("Snapshot not found: {}", snapshot_id)))?;

        let mut state: SimulationState = serde_json::from_value(snap.state_json.clone())?;
        state.id = Uuid::now_v7().to_string(); // New sim ID for the branch
        state.snapshots.clear(); // Branch starts with clean snapshot history
        Ok(state)
    }

    /// Reset actor action counts for a new turn.
    pub fn reset_turn_actions(&mut self) {
        for actor in self.actors.values_mut() {
            actor.actions_this_turn = 0;
        }
    }

    /// Advance to the next turn.
    pub fn advance_turn(&mut self) {
        self.turn_number += 1;
        self.current_time = self.current_time + Duration::minutes(self.time_step_minutes as i64);
        self.reset_turn_actions();
        self.evolve_smir();
    }

    /// Get the number of actors per team.
    pub fn team_counts(&self) -> HashMap<Team, usize> {
        let mut counts = HashMap::new();
        for actor in self.actors.values() {
            *counts.entry(actor.team).or_insert(0) += 1;
        }
        counts
    }

    /// Assign an actor to a team.
    pub fn assign_team(&mut self, entity_id: &Uuid, team: Team) -> Result<()> {
        let actor = self
            .actors
            .get_mut(entity_id)
            .ok_or_else(|| TensaError::Internal(format!("Actor not found: {}", entity_id)))?;
        actor.team = team;
        Ok(())
    }

    /// Seed misinformation into a (narrative, platform) compartment.
    pub fn seed_narrative(&mut self, narrative_id: &str, platform: &str, initial_misinformed: f64) {
        if let Some(comp) = self
            .compartments
            .get_mut(&compartment_key(narrative_id, platform))
        {
            comp.seed_misinformed(initial_misinformed);
        }
    }
}

/// Build a composite key for compartment lookups: "narrative_id|platform".
pub(crate) fn compartment_key(narrative_id: &str, platform: &str) -> String {
    format!("{}|{}", narrative_id, platform)
}

// ─── KV Storage ──────────────────────────────────────────────

const SIM_PREFIX: &[u8] = b"adv/sim/";

/// Store a simulation state in KV.
pub fn store_sim_state(hypergraph: &Hypergraph, state: &SimulationState) -> Result<()> {
    let mut key = SIM_PREFIX.to_vec();
    key.extend_from_slice(state.id.as_bytes());
    let value = serde_json::to_vec(state)?;
    hypergraph.store().put(&key, &value)?;
    Ok(())
}

/// Load a simulation state from KV.
pub fn load_sim_state(hypergraph: &Hypergraph, sim_id: &str) -> Result<Option<SimulationState>> {
    let mut key = SIM_PREFIX.to_vec();
    key.extend_from_slice(sim_id.as_bytes());
    match hypergraph.store().get(&key)? {
        Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        None => Ok(None),
    }
}

// ─── Default Platforms ───────────────────────────────────────

/// Default platform states with literature-calibrated transmission rates.
fn default_platforms() -> HashMap<String, PlatformState> {
    let mut platforms = HashMap::new();

    let defaults = [
        ("twitter", 0.4, 0.3),   // Vosoughi 2018: β ≈ 0.3-0.5
        ("facebook", 0.2, 0.15), // Post-2018 algorithm: β ≈ 0.15-0.3
        ("telegram", 0.65, 0.5), // No algorithmic suppression: β ≈ 0.5-0.8
        ("tiktok", 0.55, 0.4),   // Algorithmic amplification: β ≈ 0.4-0.7
        ("youtube", 0.15, 0.1),  // Recommendation-driven: β ≈ 0.1-0.2
    ];

    for (name, beta, _gamma) in &defaults {
        platforms.insert(
            name.to_string(),
            PlatformState {
                platform_name: name.to_string(),
                beta: *beta,
                active_accounts: 0,
                content_velocity: 0.0,
                moderation: ModerationPolicy::default(),
            },
        );
    }

    platforms
}

/// Platform state within a simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformState {
    pub platform_name: String,
    pub beta: f64,
    pub active_accounts: usize,
    pub content_velocity: f64,
    pub moderation: ModerationPolicy,
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn create_test_actor(hg: &Hypergraph, name: &str, narrative_id: &str) -> Uuid {
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
            narrative_id: Some(narrative_id.to_string()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap()
    }

    #[test]
    fn test_fork_from_hypergraph() {
        let hg = test_hg();
        let _a1 = create_test_actor(&hg, "Alice", "test-nar");
        let _a2 = create_test_actor(&hg, "Bob", "test-nar");
        let _a3 = create_test_actor(&hg, "Charlie", "other-nar");

        let state = SimulationState::fork_from_hypergraph(&hg, "test-nar", 60).unwrap();

        assert_eq!(state.actors.len(), 2, "should only include test-nar actors");
        assert_eq!(state.narrative_id, "test-nar");
        assert_eq!(state.turn_number, 0);
        assert_eq!(state.platforms.len(), 5);
        assert_eq!(state.compartments.len(), 5); // one per platform
    }

    #[test]
    fn test_smir_evolution() {
        let mut comp = SmirCompartments::new(1000.0, 0.4);
        comp.seed_misinformed(10.0);

        assert_eq!(comp.susceptible, 990.0);
        assert_eq!(comp.misinformed, 10.0);

        let r0_before = comp.r0();
        assert!(r0_before > 0.0, "R₀ should be positive after seeding");

        // Step forward
        comp.step(1.0);

        assert!(
            comp.misinformed > 10.0,
            "misinformed should grow: {}",
            comp.misinformed
        );
        assert!(comp.susceptible < 990.0, "susceptible should shrink");
        assert!(
            (comp.total() - 1000.0).abs() < 0.01,
            "total should be conserved"
        );
    }

    #[test]
    fn test_r0_computation() {
        let comp = SmirCompartments::new(1000.0, 0.4);
        // R₀ = β × S / (γ × N) = 0.4 × 1000 / (0.05 × 1000) = 8.0
        assert!(
            (comp.r0() - 8.0).abs() < 1e-10,
            "R₀ should be 8.0, got {}",
            comp.r0()
        );
    }

    #[test]
    fn test_advance_turn() {
        let hg = test_hg();
        let _a1 = create_test_actor(&hg, "Alice", "test-nar");

        let mut state = SimulationState::fork_from_hypergraph(&hg, "test-nar", 60).unwrap();
        state.seed_narrative("test-nar", "twitter", 10.0);

        let t0 = state.current_time;
        state.advance_turn();

        assert_eq!(state.turn_number, 1);
        assert!(state.current_time > t0);
    }

    #[test]
    fn test_snapshot_and_branch() {
        let hg = test_hg();
        let a1 = create_test_actor(&hg, "Alice", "test-nar");

        let mut state = SimulationState::fork_from_hypergraph(&hg, "test-nar", 60).unwrap();
        state.assign_team(&a1, Team::Red).unwrap();

        let snap_id = state.snapshot().unwrap();

        // Mutate after snapshot
        state.advance_turn();
        state.advance_turn();
        assert_eq!(state.turn_number, 2);

        // Branch from snapshot — should restore turn 0
        let branched = SimulationState::branch(&snap_id, &state.snapshots).unwrap();
        assert_eq!(branched.turn_number, 0);
        assert_ne!(branched.id, state.id, "branch should have new sim ID");

        // Verify team assignment persisted in snapshot
        let actor = branched.actors.get(&a1).unwrap();
        assert_eq!(actor.team, Team::Red);
    }

    #[test]
    fn test_branch_independence() {
        let hg = test_hg();
        let a1 = create_test_actor(&hg, "Alice", "test-nar");

        let mut state = SimulationState::fork_from_hypergraph(&hg, "test-nar", 60).unwrap();
        let snap_id = state.snapshot().unwrap();

        let mut branched = SimulationState::branch(&snap_id, &state.snapshots).unwrap();

        // Mutate branch
        branched.advance_turn();
        branched.advance_turn();

        // Original should be unaffected
        assert_eq!(state.turn_number, 0);
        assert_eq!(branched.turn_number, 2);
    }

    #[test]
    fn test_kv_persistence() {
        let hg = test_hg();
        let _a1 = create_test_actor(&hg, "Alice", "test-nar");

        let state = SimulationState::fork_from_hypergraph(&hg, "test-nar", 60).unwrap();
        let sim_id = state.id.clone();

        store_sim_state(&hg, &state).unwrap();

        let loaded = load_sim_state(&hg, &sim_id).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.id, sim_id);
        assert_eq!(loaded.narrative_id, "test-nar");
        assert_eq!(loaded.actors.len(), 1);
    }

    #[test]
    fn test_team_assignment_and_counts() {
        let hg = test_hg();
        let a1 = create_test_actor(&hg, "Alice", "test-nar");
        let a2 = create_test_actor(&hg, "Bob", "test-nar");

        let mut state = SimulationState::fork_from_hypergraph(&hg, "test-nar", 60).unwrap();
        state.assign_team(&a1, Team::Red).unwrap();
        state.assign_team(&a2, Team::Blue).unwrap();

        let counts = state.team_counts();
        assert_eq!(counts[&Team::Red], 1);
        assert_eq!(counts[&Team::Blue], 1);
    }

    #[test]
    fn test_smir_compartments_conserve_population() {
        let mut comp = SmirCompartments::new(10_000.0, 0.5);
        comp.seed_misinformed(100.0);

        for _ in 0..100 {
            comp.step(0.5);
        }

        assert!(
            (comp.total() - 10_000.0).abs() < 1.0,
            "population should be conserved: {}",
            comp.total()
        );
    }
}
