//! Skill bundles: system-prompt modules that ship with the binary.
//!
//! Phase 2 ships a single bundled skill (`studio-ui`). Phase 5 adds `tensa`
//! and `tensa-writer` bundles + a `skill_dirs` config for loading from disk.

use std::sync::Arc;

/// A skill bundle: name + description + the system-prompt body that gets
/// injected when the skill is active on a session.
#[derive(Debug, Clone)]
pub struct SkillBundle {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
}

/// Registry of bundled skills. Lookups are O(n) over a small slice; we only
/// have a handful of skills so a HashMap would be overkill.
#[derive(Debug, Default, Clone)]
pub struct SkillRegistry {
    bundles: Arc<Vec<SkillBundle>>,
}

impl SkillRegistry {
    /// Build the default registry with all bundled skills.
    /// v0.60: studio-ui. v0.64 added tensa + tensa-writer. v0.75.0 added
    /// tensa-synth (EATH synthetic generation sprint, Phase 12).
    /// v0.76.0 added tensa-reconstruction (EATH Extension Phase 15c — the
    /// inverse problem to synth generation; distinct enough workflow to
    /// warrant its own skill bundle per architect's call).
    /// v0.77.0 added tensa-opinion-dynamics (EATH Extension Phase 16c —
    /// BCM-on-hypergraphs; distinct workflow / mental model from synth +
    /// reconstruction per architect's call).
    /// v0.78.0 added tensa-fuzzy (Fuzzy Logic Sprint, Phase 13 —
    /// t-norms, aggregators, fuzzy Allen, intermediate quantifiers,
    /// graded Peterson syllogisms, fuzzy FCA, Mamdani rules, fuzzy-
    /// probabilistic hybrid). Seventh bundle.
    /// v0.79.0 added tensa-graded (Graded Acceptability & Measure
    /// Learning Sprint, Phase 6 — gradual / ranking-based argumentation
    /// semantics, ranking-supervised Choquet measure learning, ORD-Horn
    /// path-consistency closure). Eighth bundle.
    pub fn default_bundled() -> Self {
        Self {
            bundles: Arc::new(vec![
                parse_bundle("studio-ui", include_str!("../../skills/studio-ui.md")),
                parse_bundle("tensa", include_str!("../../skills/tensa.md")),
                parse_bundle("tensa-writer", include_str!("../../skills/tensa-writer.md")),
                parse_bundle("tensa-synth", include_str!("../../skills/tensa-synth.md")),
                parse_bundle(
                    "tensa-reconstruction",
                    include_str!("../../skills/tensa-reconstruction.md"),
                ),
                parse_bundle(
                    "tensa-opinion-dynamics",
                    include_str!("../../skills/tensa-opinion-dynamics.md"),
                ),
                parse_bundle("tensa-fuzzy", include_str!("../../skills/tensa-fuzzy.md")),
                parse_bundle("tensa-graded", include_str!("../../skills/tensa-graded.md")),
            ]),
        }
    }

    pub fn get(&self, name: &str) -> Option<&SkillBundle> {
        self.bundles.iter().find(|b| b.name == name)
    }

    pub fn list(&self) -> &[SkillBundle] {
        &self.bundles
    }

    /// Concatenate the system prompts of all active skills (in order) plus
    /// a small built-in preamble. Unknown skills are silently skipped — the
    /// frontend can send stale skill names without breaking the turn.
    pub fn compose_system_prompt(&self, active: &[String]) -> String {
        let mut out =
            String::from("You are the TENSA Studio assistant. Reply in clear, compact prose.\n\n");
        for name in active {
            if let Some(b) = self.get(name) {
                out.push_str(&b.system_prompt);
                out.push_str("\n\n");
            }
        }
        out
    }
}

/// Split a skill file on its `---\n…\n---` frontmatter block. The name is
/// already known (passed in from the callsite, where `include_str!` also
/// fixes it), so the parser only needs to pull out the description and the
/// body. If the file has no frontmatter we treat the whole thing as body.
fn parse_bundle(name: &str, source: &str) -> SkillBundle {
    let (description, body) = match split_frontmatter(source) {
        Some((front, rest)) => (extract_description(front), rest.to_string()),
        None => (String::new(), source.to_string()),
    };
    SkillBundle {
        name: name.to_string(),
        description,
        system_prompt: body.trim().to_string(),
    }
}

fn split_frontmatter(source: &str) -> Option<(&str, &str)> {
    let trimmed = source.trim_start_matches('\u{feff}').trim_start();
    let rest = trimmed.strip_prefix("---")?;
    let rest = rest
        .strip_prefix('\n')
        .or_else(|| rest.strip_prefix("\r\n"))?;
    let end_idx = rest.find("\n---")?;
    let front = &rest[..end_idx];
    let after = &rest[end_idx + 4..];
    let after = after.trim_start_matches('\r');
    let after = after.strip_prefix('\n').unwrap_or(after);
    Some((front, after))
}

fn extract_description(front: &str) -> String {
    for line in front.lines() {
        if let Some(rest) = line.trim().strip_prefix("description:") {
            return rest.trim().trim_matches('"').to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_studio_ui() {
        let reg = SkillRegistry::default_bundled();
        let b = reg.get("studio-ui").expect("studio-ui must be bundled");
        assert_eq!(b.name, "studio-ui");
        assert!(!b.description.is_empty(), "description should be parsed");
        assert!(
            b.system_prompt.contains("TENSA Studio"),
            "body should survive frontmatter strip"
        );
    }

    #[test]
    fn default_registry_has_eight_bundled_skills() {
        // v0.75.0 (EATH Phase 12) added tensa-synth as the fourth bundle.
        // v0.76.0 (EATH Extension Phase 15c) added tensa-reconstruction as
        // the fifth bundle — the inverse problem to synth generation.
        // v0.77.0 (EATH Extension Phase 16c) added tensa-opinion-dynamics
        // as the sixth — BCM on hypergraphs (Hickok 2022 + Schawe-Hernández
        // 2022 + Deffuant 2000).
        // v0.78.0 (Fuzzy Logic Sprint, Phase 13) added tensa-fuzzy as the
        // seventh — t-norms / aggregators / fuzzy Allen / intermediate
        // quantifiers / Peterson syllogisms / fuzzy FCA / Mamdani rules /
        // fuzzy-probabilistic hybrid.
        // v0.79.0 (Graded Acceptability & Measure Learning Sprint, Phase 6)
        // added tensa-graded as the eighth — gradual / ranking-based
        // argumentation semantics (h-Categoriser, weighted h-Categoriser,
        // max-based, card-based) / ranking-supervised Choquet measure
        // learning / ORD-Horn path-consistency closure.
        let reg = SkillRegistry::default_bundled();
        let names: Vec<&str> = reg.list().iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "studio-ui",
                "tensa",
                "tensa-writer",
                "tensa-synth",
                "tensa-reconstruction",
                "tensa-opinion-dynamics",
                "tensa-fuzzy",
                "tensa-graded",
            ]
        );
        // Spot-check tensa-synth body wires through include_str!.
        let synth = reg.get("tensa-synth").expect("tensa-synth must be bundled");
        assert!(
            synth.system_prompt.contains("EATH") || synth.system_prompt.contains("Synthetic"),
            "tensa-synth body should reference EATH / Synthetic"
        );
        assert!(
            !synth.description.is_empty(),
            "tensa-synth must have a non-empty description from frontmatter"
        );
        // Spot-check tensa-reconstruction.
        let recon = reg
            .get("tensa-reconstruction")
            .expect("tensa-reconstruction must be bundled");
        assert!(
            recon.system_prompt.contains("SINDy")
                || recon.system_prompt.contains("Reconstruction"),
            "tensa-reconstruction body should reference SINDy / Reconstruction"
        );
        assert!(
            !recon.description.is_empty(),
            "tensa-reconstruction must have a non-empty description from frontmatter"
        );
        // Spot-check tensa-opinion-dynamics.
        let opd = reg
            .get("tensa-opinion-dynamics")
            .expect("tensa-opinion-dynamics must be bundled");
        assert!(
            opd.system_prompt.contains("Hickok")
                || opd.system_prompt.contains("BCM")
                || opd.system_prompt.contains("Opinion Dynamics"),
            "tensa-opinion-dynamics body should reference Hickok / BCM / Opinion Dynamics"
        );
        assert!(
            !opd.description.is_empty(),
            "tensa-opinion-dynamics must have a non-empty description from frontmatter"
        );
        // Spot-check tensa-fuzzy.
        let fuzzy = reg.get("tensa-fuzzy").expect("tensa-fuzzy must be bundled");
        assert!(
            fuzzy.system_prompt.contains("t-norm")
                || fuzzy.system_prompt.contains("Gödel")
                || fuzzy.system_prompt.contains("aggregator"),
            "tensa-fuzzy body should reference t-norm / Gödel / aggregator"
        );
        assert!(
            !fuzzy.description.is_empty(),
            "tensa-fuzzy must have a non-empty description from frontmatter"
        );
        // Spot-check tensa-graded.
        let graded = reg
            .get("tensa-graded")
            .expect("tensa-graded must be bundled");
        assert!(
            graded.system_prompt.contains("h-Categoriser")
                || graded.system_prompt.contains("ORD-Horn")
                || graded.system_prompt.contains("Choquet"),
            "tensa-graded body should reference h-Categoriser / ORD-Horn / Choquet"
        );
        assert!(
            !graded.description.is_empty(),
            "tensa-graded must have a non-empty description from frontmatter"
        );
    }

    #[test]
    fn compose_skips_unknown_skills() {
        let reg = SkillRegistry::default_bundled();
        let s = reg.compose_system_prompt(&["studio-ui".into(), "nope".into()]);
        assert!(s.contains("TENSA Studio assistant"));
        assert!(s.contains("TENSA Studio"));
    }

    #[test]
    fn frontmatter_parser_handles_crlf() {
        let src = "---\r\nname: t\r\ndescription: hi\r\n---\r\nBody";
        let (front, body) = split_frontmatter(src).unwrap();
        assert!(front.contains("description: hi"));
        assert_eq!(body, "Body");
    }

    #[test]
    fn frontmatter_description_extracted() {
        let front = "name: t\ndescription: A short blurb";
        assert_eq!(extract_description(front), "A short blurb");
    }

    #[test]
    fn bundle_without_frontmatter_uses_entire_source_as_body() {
        let b = parse_bundle("x", "just a body");
        assert_eq!(b.system_prompt, "just a body");
        assert_eq!(b.description, "");
    }
}
