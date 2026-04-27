//! Propp's 31 narrative functions — canonical morphological classification.
//!
//! Maps situations to one of Vladimir Propp's 31 narrative functions
//! using keyword heuristics (default) or LLM classification (when available).
//! Propp functions describe the structural role a situation plays in a
//! fairy-tale / narrative morphology.

use serde::{Deserialize, Serialize};

use crate::types::Situation;

/// Vladimir Propp's 31 narrative functions.
///
/// Each function represents a canonical narrative action in morphological
/// analysis of folktales and story structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProppFunction {
    /// 1. Absentation — a family member leaves home.
    Absentation,
    /// 2. Interdiction — a warning or prohibition is given.
    Interdiction,
    /// 3. Violation — the interdiction is violated.
    Violation,
    /// 4. Reconnaissance — the villain seeks information.
    Reconnaissance,
    /// 5. Delivery — the villain learns about the hero.
    Delivery,
    /// 6. Trickery — the villain attempts deception.
    Trickery,
    /// 7. Complicity — the victim is deceived.
    Complicity,
    /// 8. Villainy — the villain harms a family member.
    Villainy,
    /// 8a. Lack — a family member lacks or desires something.
    Lack,
    /// 9. Mediation — misfortune is made known; hero is dispatched.
    Mediation,
    /// 10. CounterAction — the hero decides to act.
    CounterAction,
    /// 11. Departure — the hero leaves home.
    Departure,
    /// 12. FirstDonorFunction — the hero is tested by a donor.
    FirstDonorFunction,
    /// 13. HeroReaction — the hero responds to the donor's test.
    HeroReaction,
    /// 14. MagicalAgent — the hero receives a magical agent/helper.
    MagicalAgent,
    /// 15. Guidance — the hero is led to the object of the search.
    Guidance,
    /// 16. Struggle — hero and villain engage in combat.
    Struggle,
    /// 17. Branding — the hero is marked or branded.
    Branding,
    /// 18. Victory — the villain is defeated.
    Victory,
    /// 19. Liquidation — initial misfortune or lack is resolved.
    Liquidation,
    /// 20. Return — the hero returns.
    Return,
    /// 21. Pursuit — the hero is pursued.
    Pursuit,
    /// 22. Rescue — the hero is rescued from pursuit.
    Rescue,
    /// 23. UnrecognizedArrival — the hero arrives home unrecognized.
    UnrecognizedArrival,
    /// 24. UnfoundedClaims — a false hero presents unfounded claims.
    UnfoundedClaims,
    /// 25. DifficultTask — a difficult task is proposed to the hero.
    DifficultTask,
    /// 26. Solution — the task is solved.
    Solution,
    /// 27. Recognition — the hero is recognized.
    Recognition,
    /// 28. Exposure — the false hero or villain is exposed.
    Exposure,
    /// 29. Transfiguration — the hero is given a new appearance.
    Transfiguration,
    /// 30. Punishment — the villain is punished.
    Punishment,
    /// 31. Wedding — the hero is married or ascends the throne.
    Wedding,
}

impl ProppFunction {
    /// All 32 functions in canonical order (31 + 8a Lack).
    pub fn all() -> &'static [ProppFunction] {
        use ProppFunction::*;
        &[
            Absentation,
            Interdiction,
            Violation,
            Reconnaissance,
            Delivery,
            Trickery,
            Complicity,
            Villainy,
            Lack,
            Mediation,
            CounterAction,
            Departure,
            FirstDonorFunction,
            HeroReaction,
            MagicalAgent,
            Guidance,
            Struggle,
            Branding,
            Victory,
            Liquidation,
            Return,
            Pursuit,
            Rescue,
            UnrecognizedArrival,
            UnfoundedClaims,
            DifficultTask,
            Solution,
            Recognition,
            Exposure,
            Transfiguration,
            Punishment,
            Wedding,
        ]
    }

    /// Propp's canonical number (1-31, with 8a = 9 offset).
    pub fn number(&self) -> &'static str {
        use ProppFunction::*;
        match self {
            Absentation => "1",
            Interdiction => "2",
            Violation => "3",
            Reconnaissance => "4",
            Delivery => "5",
            Trickery => "6",
            Complicity => "7",
            Villainy => "8",
            Lack => "8a",
            Mediation => "9",
            CounterAction => "10",
            Departure => "11",
            FirstDonorFunction => "12",
            HeroReaction => "13",
            MagicalAgent => "14",
            Guidance => "15",
            Struggle => "16",
            Branding => "17",
            Victory => "18",
            Liquidation => "19",
            Return => "20",
            Pursuit => "21",
            Rescue => "22",
            UnrecognizedArrival => "23",
            UnfoundedClaims => "24",
            DifficultTask => "25",
            Solution => "26",
            Recognition => "27",
            Exposure => "28",
            Transfiguration => "29",
            Punishment => "30",
            Wedding => "31",
        }
    }

    /// Short human-readable label.
    pub fn label(&self) -> &'static str {
        use ProppFunction::*;
        match self {
            Absentation => "absentation",
            Interdiction => "interdiction",
            Violation => "violation",
            Reconnaissance => "reconnaissance",
            Delivery => "delivery",
            Trickery => "trickery",
            Complicity => "complicity",
            Villainy => "villainy",
            Lack => "lack",
            Mediation => "mediation",
            CounterAction => "counter-action",
            Departure => "departure",
            FirstDonorFunction => "first donor function",
            HeroReaction => "hero reaction",
            MagicalAgent => "magical agent",
            Guidance => "guidance",
            Struggle => "struggle",
            Branding => "branding",
            Victory => "victory",
            Liquidation => "liquidation",
            Return => "return",
            Pursuit => "pursuit",
            Rescue => "rescue",
            UnrecognizedArrival => "unrecognized arrival",
            UnfoundedClaims => "unfounded claims",
            DifficultTask => "difficult task",
            Solution => "solution",
            Recognition => "recognition",
            Exposure => "exposure",
            Transfiguration => "transfiguration",
            Punishment => "punishment",
            Wedding => "wedding",
        }
    }
}

impl std::fmt::Display for ProppFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.label(), self.number())
    }
}

impl std::str::FromStr for ProppFunction {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        use ProppFunction::*;
        match s.to_lowercase().replace(['-', '_', ' '], "").as_str() {
            "absentation" => Ok(Absentation),
            "interdiction" => Ok(Interdiction),
            "violation" => Ok(Violation),
            "reconnaissance" => Ok(Reconnaissance),
            "delivery" => Ok(Delivery),
            "trickery" => Ok(Trickery),
            "complicity" => Ok(Complicity),
            "villainy" => Ok(Villainy),
            "lack" => Ok(Lack),
            "mediation" => Ok(Mediation),
            "counteraction" => Ok(CounterAction),
            "departure" => Ok(Departure),
            "firstdonorfunction" | "donorfunction" | "testing" => Ok(FirstDonorFunction),
            "heroreaction" | "reaction" => Ok(HeroReaction),
            "magicalagent" | "receipt" | "provision" => Ok(MagicalAgent),
            "guidance" | "spatialtransference" => Ok(Guidance),
            "struggle" | "combat" => Ok(Struggle),
            "branding" | "marking" => Ok(Branding),
            "victory" => Ok(Victory),
            "liquidation" | "resolution" => Ok(Liquidation),
            "return" => Ok(Return),
            "pursuit" | "chase" => Ok(Pursuit),
            "rescue" | "deliverance" => Ok(Rescue),
            "unrecognizedarrival" => Ok(UnrecognizedArrival),
            "unfoundedclaims" | "falseclaims" => Ok(UnfoundedClaims),
            "difficulttask" | "task" => Ok(DifficultTask),
            "solution" | "tasksolved" => Ok(Solution),
            "recognition" => Ok(Recognition),
            "exposure" | "unmasking" => Ok(Exposure),
            "transfiguration" | "transformation" => Ok(Transfiguration),
            "punishment" => Ok(Punishment),
            "wedding" | "ascension" | "reward" => Ok(Wedding),
            _ => Err(format!("unknown Propp function: {s}")),
        }
    }
}

// ─── Keyword-based classifier ──────────────────────────────

/// Keyword patterns for each Propp function.
struct ProppPattern {
    function: ProppFunction,
    /// At least one of these must match for the function to be considered.
    required: &'static [&'static str],
    /// Additional keywords that boost the score.
    boost: &'static [&'static str],
}

fn propp_patterns() -> Vec<ProppPattern> {
    use ProppFunction::*;
    vec![
        ProppPattern {
            function: Absentation,
            required: &["leave", "depart", "absent", "gone", "away"],
            boost: &["home", "family", "journey"],
        },
        ProppPattern {
            function: Interdiction,
            required: &["warn", "forbid", "prohibit", "must not", "don't"],
            boost: &["danger", "rule", "command"],
        },
        ProppPattern {
            function: Violation,
            required: &["disobey", "violat", "broke", "ignored", "defied"],
            boost: &["warning", "rule", "forbidden"],
        },
        ProppPattern {
            function: Reconnaissance,
            required: &["spy", "scout", "investigat", "search", "seek"],
            boost: &["villain", "information", "discover"],
        },
        ProppPattern {
            function: Delivery,
            required: &["reveal", "learn", "discover", "found out", "told"],
            boost: &["secret", "identity", "weakness"],
        },
        ProppPattern {
            function: Trickery,
            required: &["trick", "deceiv", "disguise", "lure", "trap"],
            boost: &["false", "pretend", "scheme"],
        },
        ProppPattern {
            function: Complicity,
            required: &["fool", "deceived", "fell for", "tricked", "believed"],
            boost: &["victim", "naive", "trust"],
        },
        ProppPattern {
            function: Villainy,
            required: &["kidnap", "steal", "murder", "destroy", "harm", "attack"],
            boost: &["villain", "evil", "crime"],
        },
        ProppPattern {
            function: Lack,
            required: &["need", "want", "lack", "desire", "miss", "lost"],
            boost: &["quest", "search", "longing"],
        },
        ProppPattern {
            function: Mediation,
            required: &["dispatch", "sent", "mission", "quest", "call"],
            boost: &["hero", "task", "request"],
        },
        ProppPattern {
            function: CounterAction,
            required: &["decide", "resolve", "vow", "determine", "accept"],
            boost: &["quest", "mission", "fight"],
        },
        ProppPattern {
            function: Departure,
            required: &["set out", "embark", "journey", "travel", "depart"],
            boost: &["quest", "adventure", "road"],
        },
        ProppPattern {
            function: FirstDonorFunction,
            required: &["test", "challenge", "trial", "riddle", "prove"],
            boost: &["donor", "old", "wise", "stranger"],
        },
        ProppPattern {
            function: HeroReaction,
            required: &["pass", "answer", "succeed", "respond", "endure"],
            boost: &["test", "trial", "challenge"],
        },
        ProppPattern {
            function: MagicalAgent,
            required: &["gift", "magic", "weapon", "potion", "tool", "receive"],
            boost: &["power", "enchanted", "helper"],
        },
        ProppPattern {
            function: Guidance,
            required: &["guide", "lead", "path", "show the way", "direct"],
            boost: &["destination", "kingdom", "location"],
        },
        ProppPattern {
            function: Struggle,
            required: &["fight", "battle", "combat", "duel", "clash"],
            boost: &["villain", "enemy", "sword"],
        },
        ProppPattern {
            function: Branding,
            required: &["mark", "brand", "wound", "scar", "ring", "token"],
            boost: &["identity", "proof", "sign"],
        },
        ProppPattern {
            function: Victory,
            required: &["defeat", "vanquish", "overcome", "slay", "conquer"],
            boost: &["villain", "enemy", "triumph"],
        },
        ProppPattern {
            function: Liquidation,
            required: &["resolv", "restor", "free", "heal", "recover", "rescue"],
            boost: &["return", "peace", "cure"],
        },
        ProppPattern {
            function: Return,
            required: &["return", "come back", "homeward", "journey back"],
            boost: &["home", "kingdom", "family"],
        },
        ProppPattern {
            function: Pursuit,
            required: &["pursue", "chase", "hunt", "follow", "flee"],
            boost: &["escape", "danger", "villain"],
        },
        ProppPattern {
            function: Rescue,
            required: &["rescue", "save", "escape", "shelter", "hide"],
            boost: &["safety", "refuge", "protection"],
        },
        ProppPattern {
            function: UnrecognizedArrival,
            required: &["disguise", "unrecogniz", "incognito", "unknown"],
            boost: &["arrive", "return", "home"],
        },
        ProppPattern {
            function: UnfoundedClaims,
            required: &["false claim", "impostor", "pretend", "usurp"],
            boost: &["throne", "credit", "glory"],
        },
        ProppPattern {
            function: DifficultTask,
            required: &["task", "challenge", "ordeal", "impossible", "prove"],
            boost: &["hero", "test", "demand"],
        },
        ProppPattern {
            function: Solution,
            required: &["solve", "accomplish", "complete", "succeed", "fulfil"],
            boost: &["task", "challenge", "quest"],
        },
        ProppPattern {
            function: Recognition,
            required: &["recogni", "identif", "reveal identity", "realize"],
            boost: &["hero", "true", "mark"],
        },
        ProppPattern {
            function: Exposure,
            required: &["expos", "unmask", "reveal", "caught", "truth"],
            boost: &["villain", "false", "liar"],
        },
        ProppPattern {
            function: Transfiguration,
            required: &["transform", "new appearance", "beautiful", "crown"],
            boost: &["clothes", "form", "change"],
        },
        ProppPattern {
            function: Punishment,
            required: &["punish", "execute", "banish", "imprison", "sentence"],
            boost: &["villain", "justice", "penalty"],
        },
        ProppPattern {
            function: Wedding,
            required: &["marry", "wedding", "throne", "coronat", "reward", "reign"],
            boost: &["king", "queen", "happily"],
        },
    ]
}

/// Classify a situation into a Propp function using keyword heuristics.
///
/// Returns `None` if no function matches with sufficient confidence.
/// The `context` parameter provides surrounding narrative text for disambiguation.
pub fn classify_propp_function(
    situation: &Situation,
    context: &str,
) -> Option<(ProppFunction, f64)> {
    static PATTERNS: std::sync::OnceLock<Vec<ProppPattern>> = std::sync::OnceLock::new();
    let patterns = PATTERNS.get_or_init(propp_patterns);
    classify_with_patterns(situation, context, patterns)
}

/// Inner classifier that operates on pre-built patterns.
fn classify_with_patterns(
    situation: &Situation,
    context: &str,
    patterns: &[ProppPattern],
) -> Option<(ProppFunction, f64)> {
    let mut text = String::new();
    for block in &situation.raw_content {
        text.push_str(&block.content);
        text.push(' ');
    }
    text.push_str(context);
    let text = text.to_lowercase();

    let mut best: Option<(ProppFunction, f64)> = None;

    for pat in patterns {
        let required_hits: usize = pat.required.iter().filter(|kw| text.contains(**kw)).count();
        if required_hits == 0 {
            continue;
        }

        let boost_hits: usize = pat.boost.iter().filter(|kw| text.contains(**kw)).count();

        // Score: required hits dominate, boosts add refinement.
        let score = required_hits as f64 * 2.0 + boost_hits as f64;

        if let Some((_, best_score)) = &best {
            if score > *best_score {
                best = Some((pat.function, score));
            }
        } else {
            best = Some((pat.function, score));
        }
    }

    // Normalize score to a 0-1 confidence.
    best.map(|(func, score)| {
        let confidence = (score / 8.0).min(1.0); // 8.0 = reasonable max
        (func, confidence)
    })
}

/// Classify a sequence of situations into Propp functions.
///
/// Returns a vector of `(situation_id, Option<ProppFunction>, confidence)` tuples.
/// Situations that don't match any function get `None`.
pub fn classify_propp_sequence(
    situations: &[Situation],
) -> Vec<(uuid::Uuid, Option<ProppFunction>, f64)> {
    situations
        .iter()
        .map(|sit| match classify_propp_function(sit, "") {
            Some((func, conf)) => (sit.id, Some(func), conf),
            None => (sit.id, None, 0.0),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn make_sit(content: &str) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
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
            raw_content: vec![ContentBlock::text(content)],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: None,
            source_chunk_id: None,
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
        }
    }

    #[test]
    fn test_propp_enum_roundtrip() {
        for func in ProppFunction::all() {
            let label = func.label();
            let parsed: ProppFunction = label.parse().unwrap();
            assert_eq!(*func, parsed, "roundtrip failed for {}", label);
            // Serialization roundtrip.
            let json = serde_json::to_string(func).unwrap();
            let decoded: ProppFunction = serde_json::from_str(&json).unwrap();
            assert_eq!(*func, decoded);
        }
        assert_eq!(ProppFunction::all().len(), 32); // 31 + 8a (Lack)
    }

    #[test]
    fn test_propp_absentation() {
        let sit = make_sit(
            "The prince decided to leave home and embark on a journey far away from his family.",
        );
        let result = classify_propp_function(&sit, "");
        assert!(result.is_some());
        let (func, conf) = result.unwrap();
        assert_eq!(func, ProppFunction::Absentation);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_propp_none_ambiguous() {
        // Text with no clear Propp function keywords.
        let sit = make_sit("The sun was shining and the birds were singing on a quiet afternoon.");
        let result = classify_propp_function(&sit, "");
        assert!(result.is_none(), "Ambiguous text should return None");
    }

    #[test]
    fn test_propp_sequence() {
        let situations = vec![
            make_sit("The hero decided to leave home and depart on a quest."),
            make_sit("A wise old stranger tested the hero with a riddle."),
            make_sit("The hero fought the villain in fierce combat."),
            make_sit("The villain was defeated and vanquished."),
            make_sit("The hero married the princess in a grand wedding."),
        ];

        let seq = classify_propp_sequence(&situations);
        assert_eq!(seq.len(), 5);

        // Check some expected classifications.
        let functions: Vec<Option<ProppFunction>> = seq.iter().map(|(_, f, _)| *f).collect();
        // First should be Departure or Absentation.
        assert!(
            functions[0] == Some(ProppFunction::Departure)
                || functions[0] == Some(ProppFunction::Absentation),
            "got {:?}",
            functions[0]
        );
        // Combat should be Struggle.
        assert_eq!(functions[2], Some(ProppFunction::Struggle));
        // Defeat should be Victory.
        assert_eq!(functions[3], Some(ProppFunction::Victory));
        // Wedding.
        assert_eq!(functions[4], Some(ProppFunction::Wedding));
    }

    #[test]
    fn test_propp_villainy() {
        let sit = make_sit("The villain kidnapped the princess and destroyed the village.");
        let result = classify_propp_function(&sit, "");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, ProppFunction::Villainy);
    }

    #[test]
    fn test_propp_punishment() {
        let sit = make_sit("The wicked queen was punished and banished from the kingdom forever.");
        let result = classify_propp_function(&sit, "The villain was caught");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, ProppFunction::Punishment);
    }
}
