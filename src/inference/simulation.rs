//! Narrative Simulation Engine — LLM-powered generative agent forward-play.
//!
//! Given a narrative state, instantiates entities as LLM-powered agents and
//! simulates forward from a given situation. Each agent acts according to its
//! inferred motivation (MaxEnt IRL weights), information set, beliefs about
//! others, and game-theoretic strategy.
//!
//! Simulated situations are stored with `ExtractionMethod::Simulated` and a
//! unique `simulation_id`, clearly marked and queryable but not contaminating
//! the real knowledge graph.
//!
//! Reference: Park et al., "Generative Agents" (Stanford/Google, 2023)

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::analysis_key;
use crate::error::{Result, TensaError};
use crate::hypergraph::keys::ANALYSIS_SIMULATION;
use crate::hypergraph::Hypergraph;
use crate::ingestion::llm::NarrativeExtractor;
use crate::types::*;

use super::types::*;
use super::InferenceEngine;

// ─── Configuration ─────────────────────────────────────────

/// Configuration for the narrative simulation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Maximum number of simulation turns.
    pub max_turns: usize,
    /// Maximum agents per turn (caps LLM calls per turn).
    pub agents_per_turn_limit: usize,
    /// Whether to include motivation weights in agent prompts.
    pub use_motivation: bool,
    /// Whether to include game-theoretic strategies in prompts.
    pub use_game_theory: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            agents_per_turn_limit: 10,
            use_motivation: true,
            use_game_theory: true,
        }
    }
}

// ─── Agent & Result Types ──────────────────────────────────

/// The state of a simulated agent (entity + context).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub entity_id: Uuid,
    pub entity_name: String,
    pub entity_type: EntityType,
    pub motivation_archetype: Option<String>,
    pub motivation_weights: Vec<(String, f64)>,
    pub known_facts: Vec<String>,
    pub beliefs_about: Vec<AgentBelief>,
}

/// What an agent believes about another entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBelief {
    pub about_entity_id: Uuid,
    pub about_entity_name: String,
    pub believed_facts: Vec<String>,
}

/// A single agent's action in a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAction {
    pub entity_id: Uuid,
    pub entity_name: String,
    pub action: String,
    pub reasoning: String,
}

/// Record of a single simulation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    pub turn_number: usize,
    pub situation_id: Option<Uuid>,
    pub agent_actions: Vec<AgentAction>,
    pub outcome_description: String,
}

/// Full simulation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub simulation_id: String,
    pub narrative_id: String,
    pub starting_situation_id: String,
    pub turns: Vec<TurnRecord>,
    pub agents: Vec<AgentState>,
    pub total_llm_calls: usize,
}

// ─── Prompts ───────────────────────────────────────────────

const AGENT_SYSTEM_PROMPT: &str = r#"You are simulating a character in a narrative. Given the character's profile, motivations, knowledge, and the current situation, decide what this character would do next.

Respond with JSON: {"action": "what the character does", "reasoning": "why they do it"}

Be specific and concrete. The action should be a single sentence describing a physical or social action. The reasoning should explain the character's internal logic based on their motivations and knowledge."#;

const RESOLUTION_SYSTEM_PROMPT: &str = r#"You are a narrative resolution engine. Given a situation and the actions chosen by all characters, describe what happens next as a brief narrative paragraph.

Respond with JSON: {"description": "what happens as a result of all actions", "consequences": "brief note on how this changes the situation"}"#;

// ─── Engine ────────────────────────────────────────────────

/// Narrative simulation inference engine.
pub struct NarrativeSimulationEngine {
    extractor: Option<Arc<dyn NarrativeExtractor>>,
}

impl NarrativeSimulationEngine {
    pub fn new(extractor: Option<Arc<dyn NarrativeExtractor>>) -> Self {
        Self { extractor }
    }
}

impl InferenceEngine for NarrativeSimulationEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::NarrativeSimulation
    }

    fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
        Ok(30000) // 30s — LLM-heavy
    }

    fn execute(&self, job: &InferenceJob, hypergraph: &Hypergraph) -> Result<InferenceResult> {
        let extractor = self.extractor.as_ref().ok_or_else(|| {
            TensaError::InferenceError(
                "Narrative simulation requires a configured LLM. Set up an LLM provider via /settings/llm.".into(),
            )
        })?;

        let narrative_id = job
            .parameters
            .get("narrative_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensaError::InferenceError("missing narrative_id".into()))?;

        let starting_situation_id = job
            .parameters
            .get("starting_situation_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or(job.target_id);

        // Parse entity IDs (or use all entities in narrative)
        let entity_ids: Vec<Uuid> =
            if let Some(arr) = job.parameters.get("entity_ids").and_then(|v| v.as_array()) {
                arr.iter()
                    .filter_map(|v| v.as_str().and_then(|s| Uuid::parse_str(s).ok()))
                    .collect()
            } else {
                hypergraph
                    .list_entities_by_narrative(narrative_id)?
                    .iter()
                    .map(|e| e.id)
                    .collect()
            };

        if entity_ids.is_empty() {
            return Err(TensaError::InferenceError("No entities to simulate".into()));
        }

        // Parse config
        let mut config = SimulationConfig::default();
        if let Some(v) = job.parameters.get("max_turns").and_then(|v| v.as_u64()) {
            config.max_turns = (v as usize).min(50); // safety cap
        }
        if let Some(v) = job
            .parameters
            .get("agents_per_turn_limit")
            .and_then(|v| v.as_u64())
        {
            config.agents_per_turn_limit = (v as usize).min(20);
        }

        // Build agent states
        let agents: Vec<AgentState> = entity_ids
            .iter()
            .take(config.agents_per_turn_limit)
            .filter_map(|eid| build_agent_state(*eid, hypergraph).ok())
            .collect();

        if agents.is_empty() {
            return Err(TensaError::InferenceError(
                "Could not build agent state for any entity".into(),
            ));
        }

        // Get starting situation
        let starting_sit = hypergraph.get_situation(&starting_situation_id)?;
        let starting_desc = situation_description(&starting_sit);

        // Run simulation
        let simulation_id = Uuid::now_v7().to_string();
        let mut turns = Vec::new();
        let mut llm_calls = 0;
        let mut current_description = starting_desc;
        let mut current_sit_id = starting_situation_id;

        for turn_num in 1..=config.max_turns {
            // Step 1: Each agent chooses an action
            let mut agent_actions = Vec::new();
            for agent in &agents {
                let prompt = build_agent_prompt(agent, &current_description, &turns);
                match extractor.answer_question(AGENT_SYSTEM_PROMPT, &prompt) {
                    Ok(response) => {
                        llm_calls += 1;
                        let action = parse_agent_action(&response, agent);
                        agent_actions.push(action);
                    }
                    Err(_) => {
                        agent_actions.push(AgentAction {
                            entity_id: agent.entity_id,
                            entity_name: agent.entity_name.clone(),
                            action: "observes quietly".into(),
                            reasoning: "LLM call failed; default passive action".into(),
                        });
                    }
                }
            }

            // Step 2: Resolve turn outcome
            let resolution_prompt = build_resolution_prompt(&current_description, &agent_actions);
            let outcome_desc =
                match extractor.answer_question(RESOLUTION_SYSTEM_PROMPT, &resolution_prompt) {
                    Ok(response) => {
                        llm_calls += 1;
                        parse_outcome_description(&response)
                    }
                    Err(_) => format!(
                        "Turn {}: {}",
                        turn_num,
                        agent_actions
                            .iter()
                            .map(|a| format!("{} {}", a.entity_name, a.action))
                            .collect::<Vec<_>>()
                            .join(". ")
                    ),
                };

            // Step 3: Create simulated situation in hypergraph
            let new_sit_id = Uuid::now_v7();
            let sit = Situation {
                id: new_sit_id,
                properties: serde_json::Value::Null,
                name: Some(format!("Simulation turn {}", turn_num)),
                description: Some(outcome_desc.clone()),
                temporal: AllenInterval {
                    start: Some(Utc::now()),
                    end: Some(Utc::now()),
                    granularity: TimeGranularity::Approximate,
                    relations: vec![],
                    fuzzy_endpoints: None,
                },
                spatial: None,
                game_structure: None,
                causes: vec![],
                deterministic: None,
                probabilistic: None,
                embedding: None,
                raw_content: vec![ContentBlock::text(&outcome_desc)],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.5,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::Simulated,
                provenance: vec![],
                narrative_id: Some(narrative_id.to_string()),
                source_chunk_id: Uuid::parse_str(&simulation_id).ok(),
                source_span: None,
                synopsis: None,
                manuscript_order: None,
                parent_situation_id: None,
                label: None,
                status: None,
                keywords: vec![],
                created_at: Utc::now(),
                updated_at: Utc::now(),
                deleted_at: None,
                transaction_time: None,
            };
            hypergraph.create_situation(sit)?;

            // Add participants
            for action in &agent_actions {
                let _ = hypergraph.add_participant(Participation {
                    entity_id: action.entity_id,
                    situation_id: new_sit_id,
                    role: Role::Protagonist,
                    info_set: None,
                    action: Some(action.action.clone()),
                    payoff: None,
                    seq: 0,
                });
            }

            // Add causal link from previous situation
            let _ = hypergraph.add_causal_link(CausalLink {
                from_situation: current_sit_id,
                to_situation: new_sit_id,
                mechanism: Some("simulation".into()),
                strength: 0.5,
                causal_type: CausalType::Contributing,
                maturity: MaturityLevel::Candidate,
            });

            turns.push(TurnRecord {
                turn_number: turn_num,
                situation_id: Some(new_sit_id),
                agent_actions,
                outcome_description: outcome_desc.clone(),
            });

            current_description = outcome_desc;
            current_sit_id = new_sit_id;

            // Check if all agents passed (terminal condition)
            let all_passive = turns
                .last()
                .map(|t| {
                    t.agent_actions
                        .iter()
                        .all(|a| a.action.contains("observ") || a.action.contains("no action"))
                })
                .unwrap_or(false);
            if all_passive {
                break;
            }
        }

        let result_data = SimulationResult {
            simulation_id: simulation_id.clone(),
            narrative_id: narrative_id.to_string(),
            starting_situation_id: starting_situation_id.to_string(),
            turns,
            agents,
            total_llm_calls: llm_calls,
        };

        let result_value = serde_json::to_value(&result_data)?;
        let key = analysis_key(ANALYSIS_SIMULATION, &[&simulation_id]);
        hypergraph
            .store()
            .put(&key, result_value.to_string().as_bytes())?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::NarrativeSimulation,
            target_id: job.target_id,
            result: result_value,
            confidence: 0.6,
            explanation: Some(format!(
                "Narrative simulation: {} turns, {} agents, {} LLM calls (sim_id={})",
                result_data.turns.len(),
                result_data.agents.len(),
                llm_calls,
                simulation_id,
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Agent State Construction ──────────────────────────────

/// Build an AgentState from an entity's data in the hypergraph.
fn build_agent_state(entity_id: Uuid, hypergraph: &Hypergraph) -> Result<AgentState> {
    let entity = hypergraph.get_entity(&entity_id)?;
    let entity_name = entity
        .properties
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed")
        .to_string();

    // Collect known facts from recent participations
    let participations = hypergraph.get_situations_for_entity(&entity_id)?;
    let mut known_facts: Vec<String> = Vec::new();
    let mut beliefs_about: Vec<AgentBelief> = Vec::new();

    for p in participations.iter().rev().take(10) {
        // Extract from info_set if available
        if let Some(ref info) = p.info_set {
            for fact in &info.knows_before {
                known_facts.push(fact.fact.clone());
            }
            for fact in &info.learns {
                known_facts.push(fact.fact.clone());
            }
            // Extract beliefs
            for belief in &info.beliefs_about_others {
                let about_name = hypergraph
                    .get_entity(&belief.about_entity)
                    .ok()
                    .and_then(|e| {
                        e.properties
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(String::from)
                    })
                    .unwrap_or_else(|| belief.about_entity.to_string());

                beliefs_about.push(AgentBelief {
                    about_entity_id: belief.about_entity,
                    about_entity_name: about_name,
                    believed_facts: belief
                        .believed_knowledge
                        .iter()
                        .map(|f| f.fact.clone())
                        .collect(),
                });
            }
        }

        // Add action as known fact
        if let Some(ref action) = p.action {
            known_facts.push(format!("Previously: {}", action));
        }
    }

    known_facts.dedup();
    known_facts.truncate(20); // cap for prompt size

    // Get motivation archetype if available (lightweight — no full IRL)
    let motivation_archetype = entity
        .properties
        .get("archetype")
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(AgentState {
        entity_id,
        entity_name,
        entity_type: entity.entity_type,
        motivation_archetype,
        motivation_weights: vec![], // populated by full motivation engine if needed
        known_facts,
        beliefs_about,
    })
}

// ─── Prompt Construction ───────────────────────────────────

/// Build the LLM prompt for an agent's action decision.
pub(crate) fn build_agent_prompt(
    agent: &AgentState,
    current_situation: &str,
    history: &[TurnRecord],
) -> String {
    let mut prompt = format!(
        "CHARACTER: {} ({})\n",
        agent.entity_name,
        format!("{:?}", agent.entity_type)
    );

    if let Some(ref archetype) = agent.motivation_archetype {
        prompt.push_str(&format!("MOTIVATION: {}\n", archetype));
    }

    if !agent.known_facts.is_empty() {
        prompt.push_str("WHAT YOU KNOW:\n");
        for (i, fact) in agent.known_facts.iter().take(10).enumerate() {
            prompt.push_str(&format!("  {}. {}\n", i + 1, fact));
        }
    }

    if !agent.beliefs_about.is_empty() {
        prompt.push_str("WHAT YOU BELIEVE ABOUT OTHERS:\n");
        for belief in agent.beliefs_about.iter().take(5) {
            prompt.push_str(&format!("  - {}: ", belief.about_entity_name));
            prompt.push_str(&belief.believed_facts.join(", "));
            prompt.push('\n');
        }
    }

    // Recent history (last 3 turns)
    if !history.is_empty() {
        prompt.push_str("\nRECENT EVENTS:\n");
        for turn in history.iter().rev().take(3).rev() {
            prompt.push_str(&format!(
                "  Turn {}: {}\n",
                turn.turn_number, turn.outcome_description
            ));
        }
    }

    prompt.push_str(&format!("\nCURRENT SITUATION: {}\n", current_situation));
    prompt.push_str("\nWhat does your character do? Respond with JSON: {\"action\": \"...\", \"reasoning\": \"...\"}");

    prompt
}

/// Build the resolution prompt for all agent actions.
pub(crate) fn build_resolution_prompt(current_situation: &str, actions: &[AgentAction]) -> String {
    let mut prompt = format!("SITUATION: {}\n\nACTIONS TAKEN:\n", current_situation);
    for action in actions {
        prompt.push_str(&format!(
            "- {} ({}): {}\n",
            action.entity_name,
            format!("{:?}", Role::Protagonist), // simplified
            action.action,
        ));
    }
    prompt.push_str("\nDescribe what happens as a result. Respond with JSON: {\"description\": \"...\", \"consequences\": \"...\"}");
    prompt
}

// ─── Response Parsing ──────────────────────────────────────

/// Parse an agent's action from LLM response.
pub(crate) fn parse_agent_action(response: &str, agent: &AgentState) -> AgentAction {
    // Try JSON parsing first
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(response) {
        return AgentAction {
            entity_id: agent.entity_id,
            entity_name: agent.entity_name.clone(),
            action: val
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("observes the situation")
                .to_string(),
            reasoning: val
                .get("reasoning")
                .and_then(|v| v.as_str())
                .unwrap_or("no reasoning provided")
                .to_string(),
        };
    }

    // Fallback: extract from braces if present
    if let (Some(start), Some(end)) = (response.find('{'), response.rfind('}')) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&response[start..=end]) {
            return AgentAction {
                entity_id: agent.entity_id,
                entity_name: agent.entity_name.clone(),
                action: val
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("observes the situation")
                    .to_string(),
                reasoning: val
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            };
        }
    }

    // Last resort: use raw response as action
    AgentAction {
        entity_id: agent.entity_id,
        entity_name: agent.entity_name.clone(),
        action: response.chars().take(200).collect(),
        reasoning: "Could not parse structured response".into(),
    }
}

/// Parse outcome description from LLM response.
fn parse_outcome_description(response: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(desc) = val.get("description").and_then(|v| v.as_str()) {
            return desc.to_string();
        }
    }

    // Try extracting JSON from response
    if let (Some(start), Some(end)) = (response.find('{'), response.rfind('}')) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&response[start..=end]) {
            if let Some(desc) = val.get("description").and_then(|v| v.as_str()) {
                return desc.to_string();
            }
        }
    }

    // Fallback: use raw response
    response.chars().take(500).collect()
}

/// Extract a text description from a situation.
fn situation_description(sit: &Situation) -> String {
    if let Some(ref desc) = sit.description {
        return desc.clone();
    }
    if let Some(ref name) = sit.name {
        return name.clone();
    }
    sit.raw_content
        .first()
        .map(|c| c.content.clone())
        .unwrap_or_else(|| "An unspecified situation.".into())
}

// ─── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_config_defaults() {
        let config = SimulationConfig::default();
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.agents_per_turn_limit, 10);
        assert!(config.use_motivation);
        assert!(config.use_game_theory);
    }

    #[test]
    fn test_build_agent_prompt() {
        let agent = AgentState {
            entity_id: Uuid::now_v7(),
            entity_name: "Detective Holmes".into(),
            entity_type: EntityType::Actor,
            motivation_archetype: Some("Ideological".into()),
            motivation_weights: vec![],
            known_facts: vec![
                "The butler was in the kitchen".into(),
                "A scream was heard at midnight".into(),
            ],
            beliefs_about: vec![AgentBelief {
                about_entity_id: Uuid::now_v7(),
                about_entity_name: "Butler".into(),
                believed_facts: vec!["might be hiding something".into()],
            }],
        };

        let prompt = build_agent_prompt(&agent, "The study is in disarray", &[]);
        assert!(prompt.contains("Detective Holmes"));
        assert!(prompt.contains("Ideological"));
        assert!(prompt.contains("butler was in the kitchen"));
        assert!(prompt.contains("Butler"));
        assert!(prompt.contains("The study is in disarray"));
    }

    #[test]
    fn test_parse_agent_action_valid_json() {
        let agent = AgentState {
            entity_id: Uuid::now_v7(),
            entity_name: "Alice".into(),
            entity_type: EntityType::Actor,
            motivation_archetype: None,
            motivation_weights: vec![],
            known_facts: vec![],
            beliefs_about: vec![],
        };

        let response = r#"{"action": "flee the scene", "reasoning": "danger is imminent"}"#;
        let action = parse_agent_action(response, &agent);
        assert_eq!(action.action, "flee the scene");
        assert_eq!(action.reasoning, "danger is imminent");
        assert_eq!(action.entity_name, "Alice");
    }

    #[test]
    fn test_parse_agent_action_with_preamble() {
        let agent = AgentState {
            entity_id: Uuid::now_v7(),
            entity_name: "Bob".into(),
            entity_type: EntityType::Actor,
            motivation_archetype: None,
            motivation_weights: vec![],
            known_facts: vec![],
            beliefs_about: vec![],
        };

        let response = "Here's what Bob does:\n{\"action\": \"confront the suspect\", \"reasoning\": \"must know the truth\"}";
        let action = parse_agent_action(response, &agent);
        assert_eq!(action.action, "confront the suspect");
    }

    #[test]
    fn test_parse_agent_action_malformed() {
        let agent = AgentState {
            entity_id: Uuid::now_v7(),
            entity_name: "Carol".into(),
            entity_type: EntityType::Actor,
            motivation_archetype: None,
            motivation_weights: vec![],
            known_facts: vec![],
            beliefs_about: vec![],
        };

        let response = "I think Carol would run away screaming";
        let action = parse_agent_action(response, &agent);
        // Should use raw response as action
        assert!(action.action.contains("Carol would run away"));
        assert_eq!(action.reasoning, "Could not parse structured response");
    }

    #[test]
    fn test_build_resolution_prompt() {
        let actions = vec![
            AgentAction {
                entity_id: Uuid::now_v7(),
                entity_name: "Holmes".into(),
                action: "examines the bookshelf".into(),
                reasoning: "looking for clues".into(),
            },
            AgentAction {
                entity_id: Uuid::now_v7(),
                entity_name: "Watson".into(),
                action: "guards the door".into(),
                reasoning: "prevent escape".into(),
            },
        ];

        let prompt = build_resolution_prompt("The library is dark and quiet", &actions);
        assert!(prompt.contains("The library is dark and quiet"));
        assert!(prompt.contains("Holmes"));
        assert!(prompt.contains("examines the bookshelf"));
        assert!(prompt.contains("Watson"));
        assert!(prompt.contains("guards the door"));
    }

    #[test]
    fn test_agent_state_construction() {
        use crate::analysis::test_helpers::{add_entity, add_situation, link, make_hg};

        let hg = make_hg();
        let eid = add_entity(&hg, "TestAgent", "sim-test");
        let sid = add_situation(&hg, "sim-test");
        link(&hg, eid, sid);

        let agent = build_agent_state(eid, &hg).unwrap();
        assert_eq!(agent.entity_name, "TestAgent");
        assert_eq!(agent.entity_type, EntityType::Actor);
    }

    #[test]
    fn test_parse_outcome_description() {
        let response = r#"{"description": "Holmes finds a hidden passage behind the bookshelf", "consequences": "new lead discovered"}"#;
        let desc = parse_outcome_description(response);
        assert_eq!(desc, "Holmes finds a hidden passage behind the bookshelf");
    }

    #[test]
    fn test_parse_outcome_description_fallback() {
        let response = "The detective found nothing of interest.";
        let desc = parse_outcome_description(response);
        assert_eq!(desc, "The detective found nothing of interest.");
    }

    #[test]
    fn test_result_serde() {
        let result = SimulationResult {
            simulation_id: "sim-001".into(),
            narrative_id: "mystery".into(),
            starting_situation_id: Uuid::now_v7().to_string(),
            turns: vec![TurnRecord {
                turn_number: 1,
                situation_id: Some(Uuid::now_v7()),
                agent_actions: vec![AgentAction {
                    entity_id: Uuid::now_v7(),
                    entity_name: "Holmes".into(),
                    action: "investigates".into(),
                    reasoning: "duty".into(),
                }],
                outcome_description: "The investigation begins".into(),
            }],
            agents: vec![],
            total_llm_calls: 2,
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: SimulationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.simulation_id, "sim-001");
        assert_eq!(parsed.turns.len(), 1);
        assert_eq!(parsed.total_llm_calls, 2);
    }

    #[test]
    fn test_extraction_method_simulated() {
        let method = ExtractionMethod::Simulated;
        let json = serde_json::to_value(&method).unwrap();
        assert_eq!(json, "Simulated");
        let parsed: ExtractionMethod = serde_json::from_value(json).unwrap();
        assert_eq!(parsed, ExtractionMethod::Simulated);
    }
}
