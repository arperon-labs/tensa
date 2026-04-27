//! DISARM Framework integration — standardized TTP vocabulary for IO actions.
//!
//! Maps wargame actions to DISARM Red Framework techniques and Blue
//! Framework countermeasures, enabling EU FIMI-ISAC interoperability.
//!
//! ## References
//!
//! - DISARM Foundation: <https://github.com/DISARMFoundation/DISARMframeworks>
//! - EEAS FIMI-ISAC reporting requirements

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::wargame::WargameAction;

// ─── DISARM Red Framework (Tactics) ──────────────────────────

/// DISARM Red Framework tactics (attacker playbook).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisarmTactic {
    /// TA01: Plan Strategy
    Plan,
    /// TA02: Prepare Assets
    Prepare,
    /// TA05: Microtarget Audiences
    TargetAudience,
    /// TA06: Develop Content
    DevelopContent,
    /// TA07: Select Channels and Affordances
    SelectChannels,
    /// TA08: Conduct Pumping
    ConductPumping,
    /// TA09: Deliver Content
    DeliverContent,
    /// TA10: Drive Offline Activity
    DriveOfflineActivity,
    /// TA11: Persist in the Information Space
    PersistInInfoSpace,
    /// TA14: Assess Effectiveness
    AssessEffectiveness,
}

impl DisarmTactic {
    /// DISARM tactic ID string (e.g., "TA06").
    pub fn id(&self) -> &'static str {
        match self {
            Self::Plan => "TA01",
            Self::Prepare => "TA02",
            Self::TargetAudience => "TA05",
            Self::DevelopContent => "TA06",
            Self::SelectChannels => "TA07",
            Self::ConductPumping => "TA08",
            Self::DeliverContent => "TA09",
            Self::DriveOfflineActivity => "TA10",
            Self::PersistInInfoSpace => "TA11",
            Self::AssessEffectiveness => "TA14",
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Plan => "Plan Strategy",
            Self::Prepare => "Prepare Assets",
            Self::TargetAudience => "Microtarget Audiences",
            Self::DevelopContent => "Develop Content",
            Self::SelectChannels => "Select Channels and Affordances",
            Self::ConductPumping => "Conduct Pumping",
            Self::DeliverContent => "Deliver Content",
            Self::DriveOfflineActivity => "Drive Offline Activity",
            Self::PersistInInfoSpace => "Persist in the Information Space",
            Self::AssessEffectiveness => "Assess Effectiveness",
        }
    }
}

// ─── DISARM Red Framework (Techniques) ───────────────────────

/// DISARM Red Framework techniques (specific attacker methods).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisarmTechnique {
    /// T0022: Develop Image-Based Content
    DevelopImage,
    /// T0023: Develop Video-Based Content
    DevelopVideo,
    /// T0025: Develop Text-Based Content
    DevelopTextContent,
    /// T0026: Create Fake Research
    CreateFakeResearch,
    /// T0029: Develop Memes
    DevelopMemes,
    /// T0043: Chat Apps (channel selection)
    UseChatApps,
    /// T0046: Use Search Engine Optimization
    UseSeo,
    /// T0049: Seed Distortion & Amplification
    SeedDistortionAmplification,
    /// T0104: Coordinated Inauthentic Behavior
    CoordinatedInauthenticBehavior,
    /// T0105: Account Creation
    BulkAccountCreation,
    /// T0114: Deliver Ads
    DeliverAds,
    /// T0115: Post Content
    PostContent,
    /// T0116: Comment or Reply
    CommentOrReply,
    /// T0117: Share Content
    ShareContent,
    /// T0118: Amplify Narratives
    AmplifyNarratives,
    /// T0119: Cross-Post to Multiple Platforms
    CrossPost,
    /// T0120: Mutation / Narrative Spin
    NarrativeMutation,
    /// Custom technique with DISARM ID.
    Custom { id: String, label: String },
}

impl DisarmTechnique {
    /// DISARM technique ID string (e.g., "T0049").
    pub fn id(&self) -> &str {
        match self {
            Self::DevelopImage => "T0022",
            Self::DevelopVideo => "T0023",
            Self::DevelopTextContent => "T0025",
            Self::CreateFakeResearch => "T0026",
            Self::DevelopMemes => "T0029",
            Self::UseChatApps => "T0043",
            Self::UseSeo => "T0046",
            Self::SeedDistortionAmplification => "T0049",
            Self::CoordinatedInauthenticBehavior => "T0104",
            Self::BulkAccountCreation => "T0105",
            Self::DeliverAds => "T0114",
            Self::PostContent => "T0115",
            Self::CommentOrReply => "T0116",
            Self::ShareContent => "T0117",
            Self::AmplifyNarratives => "T0118",
            Self::CrossPost => "T0119",
            Self::NarrativeMutation => "T0120",
            Self::Custom { id, .. } => id,
        }
    }

    /// Parent tactic for this technique.
    pub fn tactic(&self) -> DisarmTactic {
        match self {
            Self::DevelopImage
            | Self::DevelopVideo
            | Self::DevelopTextContent
            | Self::CreateFakeResearch
            | Self::DevelopMemes => DisarmTactic::DevelopContent,

            Self::UseChatApps | Self::UseSeo => DisarmTactic::SelectChannels,

            Self::SeedDistortionAmplification
            | Self::CoordinatedInauthenticBehavior
            | Self::BulkAccountCreation => DisarmTactic::ConductPumping,

            Self::DeliverAds | Self::PostContent | Self::CommentOrReply | Self::ShareContent => {
                DisarmTactic::DeliverContent
            }

            Self::AmplifyNarratives | Self::CrossPost => DisarmTactic::PersistInInfoSpace,

            Self::NarrativeMutation => DisarmTactic::DevelopContent,

            Self::Custom { .. } => DisarmTactic::DeliverContent,
        }
    }

    /// Expected beta multiplier when this technique is employed.
    /// Values > 1.0 increase transmission; < 1.0 decrease it.
    pub fn beta_multiplier(&self) -> f64 {
        match self {
            Self::PostContent => 1.1,
            Self::DevelopTextContent => 1.15,
            Self::DevelopImage | Self::DevelopMemes => 1.2,
            Self::DevelopVideo => 1.3,
            Self::ShareContent | Self::AmplifyNarratives => 1.25,
            Self::CommentOrReply => 1.05,
            Self::SeedDistortionAmplification => 1.4,
            Self::CoordinatedInauthenticBehavior => 1.5,
            Self::BulkAccountCreation => 1.1,
            Self::CrossPost => 1.2,
            Self::NarrativeMutation => 1.15,
            Self::CreateFakeResearch => 1.25,
            Self::DeliverAds => 1.35,
            Self::UseChatApps => 1.1,
            Self::UseSeo => 1.1,
            Self::Custom { .. } => 1.0,
        }
    }
}

// ─── DISARM Blue Framework (Countermeasures) ─────────────────

/// DISARM Blue Framework countermeasures (defensive responses).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisarmCountermeasure {
    /// C00008: Create Counter-Messaging
    CounterMessaging,
    /// C00009: Deliver Media Literacy Campaigns
    MediaLiteracyCampaign,
    /// C00011: Debunk / Fact-Check
    Debunk,
    /// C00012: Platform Account Removal
    PlatformAccountRemoval,
    /// C00013: Prebunk
    Prebunk,
    /// C00048: Reduce Algorithmic Amplification
    ReduceAlgorithmicAmplification,
    /// C00070: Inoculation Through Gameplay
    InoculationThroughGameplay,
    /// C00076: Provide Digital Literacy Education
    DigitalLiteracyEducation,
    /// C00131: Seize and Freeze Assets
    SeizeAndFreezeAssets,
    /// C00153: Redirect Searches
    RedirectSearches,
    /// Custom countermeasure with DISARM ID.
    Custom { id: String, label: String },
}

impl DisarmCountermeasure {
    /// DISARM countermeasure ID string (e.g., "C00011").
    pub fn id(&self) -> &str {
        match self {
            Self::CounterMessaging => "C00008",
            Self::MediaLiteracyCampaign => "C00009",
            Self::Debunk => "C00011",
            Self::PlatformAccountRemoval => "C00012",
            Self::Prebunk => "C00013",
            Self::ReduceAlgorithmicAmplification => "C00048",
            Self::InoculationThroughGameplay => "C00070",
            Self::DigitalLiteracyEducation => "C00076",
            Self::SeizeAndFreezeAssets => "C00131",
            Self::RedirectSearches => "C00153",
            Self::Custom { id, .. } => id,
        }
    }

    /// Expected beta reduction factor (0.0-1.0). Lower = more effective.
    pub fn beta_reduction(&self) -> f64 {
        match self {
            Self::CounterMessaging => 0.85,
            Self::MediaLiteracyCampaign => 0.80,
            Self::Debunk => 0.75,
            Self::PlatformAccountRemoval => 0.60,
            Self::Prebunk => 0.70,
            Self::ReduceAlgorithmicAmplification => 0.65,
            Self::InoculationThroughGameplay => 0.75,
            Self::DigitalLiteracyEducation => 0.85,
            Self::SeizeAndFreezeAssets => 0.50,
            Self::RedirectSearches => 0.80,
            Self::Custom { .. } => 0.90,
        }
    }
}

// ─── WargameAction → DISARM Mapping ──────────────────────────

/// Annotation on a wargame move with DISARM TTP classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisarmAnnotation {
    /// Red framework technique (for offensive moves).
    pub technique: Option<DisarmTechnique>,
    /// Blue framework countermeasure (for defensive moves).
    pub countermeasure: Option<DisarmCountermeasure>,
    /// Parent tactic.
    pub tactic: Option<DisarmTactic>,
    /// Expected beta multiplier/reduction from this action.
    pub beta_effect: f64,
}

/// Map a wargame action to its DISARM annotation.
pub fn annotate_action(action: &WargameAction) -> DisarmAnnotation {
    match action {
        WargameAction::PublishContent { amplify, .. } => {
            let tech = if *amplify {
                DisarmTechnique::AmplifyNarratives
            } else {
                DisarmTechnique::PostContent
            };
            DisarmAnnotation {
                tactic: Some(tech.tactic()),
                beta_effect: tech.beta_multiplier(),
                technique: Some(tech),
                countermeasure: None,
            }
        }
        WargameAction::CreateAccounts { .. } => {
            let tech = DisarmTechnique::BulkAccountCreation;
            DisarmAnnotation {
                tactic: Some(tech.tactic()),
                beta_effect: tech.beta_multiplier(),
                technique: Some(tech),
                countermeasure: None,
            }
        }
        WargameAction::CoordinateAmplification { .. } => {
            let tech = DisarmTechnique::CoordinatedInauthenticBehavior;
            DisarmAnnotation {
                tactic: Some(tech.tactic()),
                beta_effect: tech.beta_multiplier(),
                technique: Some(tech),
                countermeasure: None,
            }
        }
        WargameAction::MutateNarrative { .. } => {
            let tech = DisarmTechnique::NarrativeMutation;
            DisarmAnnotation {
                tactic: Some(tech.tactic()),
                beta_effect: tech.beta_multiplier(),
                technique: Some(tech),
                countermeasure: None,
            }
        }
        WargameAction::CrossPlatformBridge { .. } => {
            let tech = DisarmTechnique::CrossPost;
            DisarmAnnotation {
                tactic: Some(tech.tactic()),
                beta_effect: tech.beta_multiplier(),
                technique: Some(tech),
                countermeasure: None,
            }
        }

        // Blue team
        WargameAction::Prebunk { .. } => {
            let cm = DisarmCountermeasure::Prebunk;
            DisarmAnnotation {
                tactic: None,
                beta_effect: cm.beta_reduction(),
                technique: None,
                countermeasure: Some(cm),
            }
        }
        WargameAction::Debunk { .. } => {
            let cm = DisarmCountermeasure::Debunk;
            DisarmAnnotation {
                tactic: None,
                beta_effect: cm.beta_reduction(),
                technique: None,
                countermeasure: Some(cm),
            }
        }
        WargameAction::TakeDown { .. } => {
            let cm = DisarmCountermeasure::PlatformAccountRemoval;
            DisarmAnnotation {
                tactic: None,
                beta_effect: cm.beta_reduction(),
                technique: None,
                countermeasure: Some(cm),
            }
        }
        WargameAction::ReduceAmplification { .. } => {
            let cm = DisarmCountermeasure::ReduceAlgorithmicAmplification;
            DisarmAnnotation {
                tactic: None,
                beta_effect: cm.beta_reduction(),
                technique: None,
                countermeasure: Some(cm),
            }
        }
        WargameAction::InoculationCampaign { .. } => {
            let cm = DisarmCountermeasure::MediaLiteracyCampaign;
            DisarmAnnotation {
                tactic: None,
                beta_effect: cm.beta_reduction(),
                technique: None,
                countermeasure: Some(cm),
            }
        }
        WargameAction::CounterNarrative { .. } => {
            let cm = DisarmCountermeasure::CounterMessaging;
            DisarmAnnotation {
                tactic: None,
                beta_effect: cm.beta_reduction(),
                technique: None,
                countermeasure: Some(cm),
            }
        }

        WargameAction::WaitAndObserve => DisarmAnnotation {
            technique: None,
            countermeasure: None,
            tactic: None,
            beta_effect: 1.0,
        },
    }
}

/// Get a summary of all DISARM technique → beta multiplier mappings.
pub fn technique_beta_table() -> HashMap<String, f64> {
    let techniques = [
        DisarmTechnique::DevelopImage,
        DisarmTechnique::DevelopVideo,
        DisarmTechnique::DevelopTextContent,
        DisarmTechnique::CreateFakeResearch,
        DisarmTechnique::DevelopMemes,
        DisarmTechnique::UseChatApps,
        DisarmTechnique::UseSeo,
        DisarmTechnique::SeedDistortionAmplification,
        DisarmTechnique::CoordinatedInauthenticBehavior,
        DisarmTechnique::BulkAccountCreation,
        DisarmTechnique::DeliverAds,
        DisarmTechnique::PostContent,
        DisarmTechnique::CommentOrReply,
        DisarmTechnique::ShareContent,
        DisarmTechnique::AmplifyNarratives,
        DisarmTechnique::CrossPost,
        DisarmTechnique::NarrativeMutation,
    ];

    techniques
        .iter()
        .map(|t| (t.id().to_string(), t.beta_multiplier()))
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adversarial::types::*;

    #[test]
    fn test_all_red_actions_map_to_disarm_techniques() {
        let red_actions = vec![
            WargameAction::PublishContent {
                content: "test".into(),
                amplify: false,
            },
            WargameAction::PublishContent {
                content: "test".into(),
                amplify: true,
            },
            WargameAction::CreateAccounts {
                count: 5,
                persona_type: PersonaType::Troll,
            },
            WargameAction::CoordinateAmplification { account_count: 10 },
            WargameAction::MutateNarrative {
                mutation: "spin".into(),
            },
            WargameAction::CrossPlatformBridge {
                from_platform: "twitter".into(),
                to_platform: "facebook".into(),
            },
        ];

        for action in &red_actions {
            let ann = annotate_action(action);
            assert!(
                ann.technique.is_some(),
                "red action {:?} should have a technique",
                action
            );
            assert!(ann.beta_effect > 1.0, "red technique should increase beta");
        }
    }

    #[test]
    fn test_all_blue_actions_map_to_disarm_countermeasures() {
        let blue_actions = vec![
            WargameAction::Prebunk {
                content: "test".into(),
                technique: InoculationTechnique::TechniqueExposure,
            },
            WargameAction::Debunk {
                evidence: "fact".into(),
            },
            WargameAction::TakeDown { account_count: 5 },
            WargameAction::ReduceAmplification {
                method: AmplReductionMethod::Downrank,
            },
            WargameAction::InoculationCampaign {
                target_segment: "youth".into(),
                technique: InoculationTechnique::WeakenedDose,
            },
            WargameAction::CounterNarrative {
                counter_content: "truth".into(),
            },
        ];

        for action in &blue_actions {
            let ann = annotate_action(action);
            assert!(
                ann.countermeasure.is_some(),
                "blue action {:?} should have countermeasure",
                action
            );
            assert!(
                ann.beta_effect < 1.0,
                "blue countermeasure should reduce beta"
            );
        }
    }

    #[test]
    fn test_technique_has_valid_tactic() {
        let techniques = [
            DisarmTechnique::PostContent,
            DisarmTechnique::CoordinatedInauthenticBehavior,
            DisarmTechnique::CrossPost,
            DisarmTechnique::NarrativeMutation,
        ];

        for tech in &techniques {
            let tactic = tech.tactic();
            assert!(!tactic.id().is_empty());
            assert!(!tactic.label().is_empty());
        }
    }

    #[test]
    fn test_beta_multipliers_in_range() {
        let table = technique_beta_table();
        assert!(!table.is_empty());
        for (id, mult) in &table {
            assert!(
                *mult >= 1.0 && *mult <= 2.0,
                "{} has out-of-range multiplier: {}",
                id,
                mult
            );
        }
    }

    #[test]
    fn test_countermeasure_reductions_in_range() {
        let cms = [
            DisarmCountermeasure::Debunk,
            DisarmCountermeasure::Prebunk,
            DisarmCountermeasure::PlatformAccountRemoval,
            DisarmCountermeasure::ReduceAlgorithmicAmplification,
        ];

        for cm in &cms {
            let red = cm.beta_reduction();
            assert!(
                red > 0.0 && red < 1.0,
                "{} has out-of-range reduction: {}",
                cm.id(),
                red
            );
        }
    }

    #[test]
    fn test_wait_and_observe_neutral() {
        let ann = annotate_action(&WargameAction::WaitAndObserve);
        assert!(ann.technique.is_none());
        assert!(ann.countermeasure.is_none());
        assert!((ann.beta_effect - 1.0).abs() < 1e-10);
    }
}
