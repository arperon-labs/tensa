//! MAVEN-ERE dataset loader (event relation extraction).
//!
//! Format: JSONL where each line is a document with events and event-event relations.
//! Relations are stored as dicts mapping relation_type → list of [from_id, to_id] pairs.
//!
//! Temporal relation types: BEFORE, OVERLAP, CONTAINS, SIMULTANEOUS, ENDS-ON, BEGINS-ON
//! Causal relation types: CAUSE, PRECONDITION
//! Subevent: list of [parent, child] pairs
//!
//! Source: https://github.com/THU-KEG/MAVEN-ERE
//! License: CC-BY-SA 4.0

use super::{DatasetLoader, Split};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// A MAVEN-ERE document with events and event-event relations.
#[derive(Debug, Clone)]
pub struct MavenDocument {
    pub id: String,
    pub title: String,
    pub text: String,
    pub events: Vec<MavenEvent>,
    pub temporal_relations: Vec<MavenRelation>,
    pub causal_relations: Vec<MavenRelation>,
    pub subevent_relations: Vec<MavenRelation>,
}

/// An event in a MAVEN-ERE document.
#[derive(Debug, Clone)]
pub struct MavenEvent {
    pub id: String,
    pub trigger_word: String,
    pub event_type: String,
    pub sent_id: usize,
}

/// An event-event relation.
#[derive(Debug, Clone)]
pub struct MavenRelation {
    pub from_event: String,
    pub to_event: String,
    pub relation_type: String,
}

// ── Raw deserialization types ──

#[derive(Deserialize)]
struct RawDoc {
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    tokens: Vec<Vec<String>>,
    #[serde(default)]
    sentences: Vec<RawSentence>,
    #[serde(default)]
    events: Vec<RawEvent>,
    #[serde(default)]
    temporal_relations: RawRelationDict,
    #[serde(default)]
    causal_relations: RawRelationDict,
    #[serde(default)]
    subevent_relations: RawSubeventRelations,
}

#[derive(Deserialize, Default)]
struct RawSentence {
    #[serde(default)]
    sentence: String,
    #[serde(default)]
    tokens: Vec<String>,
}

#[derive(Deserialize)]
struct RawEvent {
    id: String,
    #[serde(default)]
    mention: Vec<RawMention>,
    #[serde(default, rename = "type")]
    event_type: String,
}

#[derive(Deserialize)]
struct RawMention {
    #[serde(default)]
    trigger_word: String,
    #[serde(default)]
    sent_id: usize,
    #[serde(default)]
    offset: Vec<usize>,
}

/// Relations stored as {"BEFORE": [[from, to], ...], "OVERLAP": [...], ...}
#[derive(Deserialize, Default)]
#[serde(transparent)]
struct RawRelationDict(HashMap<String, Vec<(String, String)>>);

/// Subevent relations: can be a list of pairs or a dict.
#[derive(Deserialize, Default)]
#[serde(transparent)]
struct RawSubeventRelations(
    #[serde(deserialize_with = "deserialize_subevent")] Vec<(String, String)>,
);

fn deserialize_subevent<'de, D>(deserializer: D) -> Result<Vec<(String, String)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct SubeventVisitor;

    impl<'de> de::Visitor<'de> for SubeventVisitor {
        type Value = Vec<(String, String)>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a list of pairs or a dict of relation_type -> pairs")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut pairs = Vec::new();
            while let Some(pair) = seq.next_element::<(String, String)>()? {
                pairs.push(pair);
            }
            Ok(pairs)
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: de::MapAccess<'de>,
        {
            let mut pairs = Vec::new();
            while let Some((_, v)) = map.next_entry::<String, Vec<(String, String)>>()? {
                pairs.extend(v);
            }
            Ok(pairs)
        }
    }

    deserializer.deserialize_any(SubeventVisitor)
}

pub struct MavenEre;

impl DatasetLoader for MavenEre {
    type Item = MavenDocument;

    fn load(data_dir: &Path, split: Split) -> Result<Vec<Self::Item>, String> {
        let dataset_dir = data_dir.join(Self::dir_name());

        let filename = format!("{}.jsonl", split.filename_suffix());
        let path = dataset_dir.join(&filename);

        if !path.exists() {
            return Err(format!(
                "MAVEN-ERE {} not found at {}",
                filename,
                path.display()
            ));
        }

        load_jsonl(&path)
    }

    fn dir_name() -> &'static str {
        "maven_ere"
    }
}

fn load_jsonl(path: &Path) -> Result<Vec<MavenDocument>, String> {
    let raw: Vec<RawDoc> = super::load_jsonl(path)?;
    Ok(raw.into_iter().map(convert_raw).collect())
}

fn convert_raw(raw: RawDoc) -> MavenDocument {
    // Build text from tokens or sentences
    let text = if !raw.tokens.is_empty() {
        raw.tokens
            .iter()
            .map(|sent| sent.join(" "))
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        raw.sentences
            .iter()
            .map(|s| {
                if !s.sentence.is_empty() {
                    s.sentence.clone()
                } else {
                    s.tokens.join(" ")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    };

    // Extract events — use first mention of each event
    let events: Vec<MavenEvent> = raw
        .events
        .iter()
        .map(|e| {
            let (trigger, sent_id) = e
                .mention
                .first()
                .map(|m| (m.trigger_word.clone(), m.sent_id))
                .unwrap_or_default();
            MavenEvent {
                id: e.id.clone(),
                trigger_word: trigger,
                event_type: e.event_type.clone(),
                sent_id,
            }
        })
        .collect();

    // Convert relation dicts to flat lists
    let temporal_relations = flatten_relation_dict(&raw.temporal_relations.0);
    let causal_relations = flatten_relation_dict(&raw.causal_relations.0);
    let subevent_relations = raw
        .subevent_relations
        .0
        .into_iter()
        .map(|(from, to)| MavenRelation {
            from_event: from,
            to_event: to,
            relation_type: "SUBEVENT".to_string(),
        })
        .collect();

    MavenDocument {
        id: raw.id,
        title: raw.title,
        text,
        events,
        temporal_relations,
        causal_relations,
        subevent_relations,
    }
}

/// Flatten {"BEFORE": [[a,b], ...], "OVERLAP": [...]} into Vec<MavenRelation>.
fn flatten_relation_dict(dict: &HashMap<String, Vec<(String, String)>>) -> Vec<MavenRelation> {
    let mut relations = Vec::new();
    for (rel_type, pairs) in dict {
        for (from, to) in pairs {
            relations.push(MavenRelation {
                from_event: from.clone(),
                to_event: to.clone(),
                relation_type: rel_type.clone(),
            });
        }
    }
    relations
}

impl MavenDocument {
    /// Get all relations (temporal + causal + subevent) in a flat list.
    pub fn all_relations(&self) -> Vec<&MavenRelation> {
        self.temporal_relations
            .iter()
            .chain(self.causal_relations.iter())
            .chain(self.subevent_relations.iter())
            .collect()
    }

    /// Count of relations by type.
    pub fn relation_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for r in self.all_relations() {
            *counts.entry(r.relation_type.clone()).or_insert(0) += 1;
        }
        counts
    }
}
