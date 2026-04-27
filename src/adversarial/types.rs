//! Shared types for the adversarial narrative wargaming module.
//!
//! Defines teams, constraints, action timing, platforms, and other
//! foundational types used across policy generation, simulation,
//! wargaming, and counter-narrative pipelines.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export the crate-level Platform to avoid duplication.
pub use crate::types::Platform;

// ─── Teams ───────────────────────────────────────────────────

/// Team assignment for wargame participants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Team {
    /// Adversary / attacker team.
    Red,
    /// Defender / analyst team.
    Blue,
    /// Organic / unaffiliated population.
    Grey,
}

// ─── Operational Constraints ─────────────────────────────────

/// Realistic constraints preventing superhuman adversary play.
///
/// Based on observed operational patterns from IRA (Linvill & Warren 2019),
/// Doppelganger (EU DisinfoLab 2022), and Spamouflage (Graphika 2019).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalConstraints {
    /// Maximum posts per day across all platforms.
    pub budget_posts_per_day: usize,
    /// Platforms the actor can operate on.
    pub platforms: Vec<Platform>,
    /// Working hours in UTC (start_hour, end_hour). None = 24/7.
    pub working_hours: Option<(u8, u8)>,
    /// Operational security level: 0.0 = sloppy, 1.0 = perfect.
    pub opsec_level: f64,
    /// Content generation quality: 0.0 = copy-paste, 1.0 = native quality.
    pub content_generation_quality: f64,
    /// Coordination latency between actors in minutes.
    pub coordination_latency_minutes: f64,
}

impl Default for OperationalConstraints {
    fn default() -> Self {
        Self {
            budget_posts_per_day: 50,
            platforms: vec![Platform::Twitter, Platform::Facebook],
            working_hours: Some((9, 18)),
            opsec_level: 0.5,
            content_generation_quality: 0.5,
            coordination_latency_minutes: 30.0,
        }
    }
}

// ─── Action Timing ───────────────────────────────────────────

/// When an adversary action should be executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionTiming {
    /// Execute immediately.
    Immediate,
    /// Delay by the specified number of minutes.
    DelayMinutes(f64),
    /// Execute at the optimal time determined by the model.
    Optimal,
    /// Execute during the specified hour range (UTC).
    WindowHours(u8, u8),
}

// ─── Action Types ────────────────────────────────────────────

/// Adversary action in the information environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryAction {
    /// Type of action to take.
    pub action_type: ActionType,
    /// Target narrative identifier.
    pub target_narrative: String,
    /// Target platform for the action.
    pub target_platform: Platform,
    /// Content guidance or template.
    pub content_template: String,
    /// When to execute.
    pub timing: ActionTiming,
    /// Expected reward under the actor's utility function.
    pub expected_reward: f64,
    /// Confidence in this action's effectiveness.
    pub confidence: f64,
}

/// Enumeration of available actions in the information environment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Publish original content on a platform.
    Post,
    /// Amplify existing content (retweet, share, etc.).
    Amplify,
    /// Reply to or quote existing content.
    Reply,
    /// Create new accounts or personas.
    CreateAccount,
    /// Coordinate amplification across multiple accounts.
    Coordinate,
    /// Publish a debunking or fact-check.
    Debunk,
    /// Publish a prebunking / inoculation message.
    Prebunk,
    /// Request platform takedown of content or accounts.
    TakeDown,
    /// Wait and observe without acting.
    Observe,
}

impl std::fmt::Display for ActionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Post => write!(f, "post"),
            Self::Amplify => write!(f, "amplify"),
            Self::Reply => write!(f, "reply"),
            Self::CreateAccount => write!(f, "create_account"),
            Self::Coordinate => write!(f, "coordinate"),
            Self::Debunk => write!(f, "debunk"),
            Self::Prebunk => write!(f, "prebunk"),
            Self::TakeDown => write!(f, "takedown"),
            Self::Observe => write!(f, "observe"),
        }
    }
}

// ─── Persona Types ───────────────────────────────────────────

/// Type of online persona for account creation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PersonaType {
    /// News aggregator account.
    NewsFeed,
    /// Political commentary account.
    PoliticalCommentator,
    /// Hashtag activism account.
    HashtagGamer,
    /// Fear-mongering account.
    Fearmonger,
    /// Local community account.
    LocalCommunity,
    /// Generic troll account.
    Troll,
}

// ─── Inoculation Techniques ──────────────────────────────────

/// Inoculation technique for prebunking campaigns.
/// Based on Roozenbeek, van der Linden et al. (2022).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InoculationTechnique {
    /// Explain the manipulation technique being used.
    TechniqueExposure,
    /// Present a weakened dose of the misinformation.
    WeakenedDose,
    /// Lateral reading exercise.
    LateralReading,
    /// Source credibility assessment.
    SourceCredibility,
}

// ─── Amplification Reduction Methods ─────────────────────────

/// Method for reducing algorithmic amplification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AmplReductionMethod {
    /// Reduce content ranking in platform algorithm.
    Downrank,
    /// Add interstitial warning labels.
    Label,
    /// Remove from recommendation feeds.
    RemoveFromRecommendations,
    /// Limit sharing / forwarding.
    LimitSharing,
}

// ─── Moderation Policy ───────────────────────────────────────

/// Platform content moderation policy model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationPolicy {
    /// Probability of takedown per reported account per day.
    pub takedown_probability: f64,
    /// Average time to takedown in hours.
    pub takedown_delay_hours: f64,
    /// Whether the platform has CIB detection.
    pub has_cib_detection: bool,
    /// Algorithmic amplification factor (1.0 = neutral).
    pub amplification_factor: f64,
}

impl Default for ModerationPolicy {
    fn default() -> Self {
        Self {
            takedown_probability: 0.1,
            takedown_delay_hours: 48.0,
            has_cib_detection: false,
            amplification_factor: 1.0,
        }
    }
}

// ─── Simulation Identifiers ──────────────────────────────────

/// Unique identifier for a simulation session.
pub type SimulationId = String;

/// Unique identifier for a narrative within a simulation.
pub type NarrativeId = String;

/// Unique identifier for a claim.
pub type ClaimId = String;

/// Entity identifier (UUID).
pub type EntityId = Uuid;
