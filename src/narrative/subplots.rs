//! Subplot detection and architecture analysis (Sprint D9.4).
//!
//! Meso-scale structural analysis: detect semi-independent storylines via
//! community detection on the situation interaction graph, classify their
//! relationship to the main plot, and measure convergence patterns.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::hypergraph::Hypergraph;
use crate::inference::InferenceEngine;
use crate::types::InferenceResult;
use crate::types::{InferenceJobType, JobStatus, Role};

// ─── Types ──────────────────────────────────────────────────

/// How a subplot relates to the main plot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubplotRelation {
    /// Echoes main theme in different context.
    Mirror,
    /// Inverts main theme.
    Contrast,
    /// Creates obstacles for main plot.
    Complication,
    /// Eventually merges into main climax.
    Convergence,
    /// Parallel story, thematic connection only.
    Independent,
    /// Exists to set up a main plot payoff.
    Setup,
}

/// A detected subplot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subplot {
    pub id: Uuid,
    pub narrative_id: String,
    pub label: String,
    /// Situations belonging to this subplot.
    pub situations: Vec<Uuid>,
    /// Characters primarily in this subplot.
    pub characters: Vec<Uuid>,
    /// Active chapter range.
    pub start_chapter: usize,
    pub end_chapter: usize,
    /// Relationship to the main plot.
    pub relation_to_main: SubplotRelation,
    /// Where this subplot merges into the main climax (if convergent).
    pub convergence_point: Option<Uuid>,
}

/// Subplot analysis results for a narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubplotAnalysis {
    pub narrative_id: String,
    pub subplots: Vec<Subplot>,
    /// Number of active subplots per chapter.
    pub density_per_chapter: Vec<usize>,
    /// Average subplot density across all chapters.
    pub average_density: f64,
    /// Fraction of subplots that converge into the main climax.
    pub convergence_ratio: f64,
}

// ─── KV ─────────────────────────────────────────────────────

fn subplot_key(narrative_id: &str) -> Vec<u8> {
    format!("sp/{}", narrative_id).into_bytes()
}

pub fn store_subplot_analysis(hg: &Hypergraph, analysis: &SubplotAnalysis) -> Result<()> {
    let key = subplot_key(&analysis.narrative_id);
    let val = serde_json::to_vec(analysis)?;
    hg.store().put(&key, &val)
}

pub fn load_subplot_analysis(
    hg: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<SubplotAnalysis>> {
    let key = subplot_key(narrative_id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

// ─── Detection ──────────────────────────────────────────────

/// Detect subplots by community detection on the situation interaction graph.
///
/// Build a graph where situations are nodes and edges are weighted by shared
/// entities. Use label propagation to find communities. The community containing
/// the protagonist is the main plot; others are subplots.
pub fn detect_subplots(hg: &Hypergraph, narrative_id: &str) -> Result<SubplotAnalysis> {
    let situations = hg.list_situations_by_narrative(narrative_id)?;
    if situations.is_empty() {
        return Ok(SubplotAnalysis {
            narrative_id: narrative_id.to_string(),
            subplots: Vec::new(),
            density_per_chapter: Vec::new(),
            average_density: 0.0,
            convergence_ratio: 0.0,
        });
    }

    // Sort situations temporally → chapter assignment
    let mut sorted_sits: Vec<_> = situations.iter().collect();
    sorted_sits.sort_by(|a, b| a.temporal.start.cmp(&b.temporal.start));
    let chapter_map: HashMap<Uuid, usize> = sorted_sits
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i))
        .collect();

    // Build situation-to-entities map + collect protagonists in single pass
    let mut sit_entities: HashMap<Uuid, HashSet<Uuid>> = HashMap::new();
    let mut protagonists: HashSet<Uuid> = HashSet::new();
    for sit in &situations {
        let participants = hg.get_participants_for_situation(&sit.id)?;
        for p in &participants {
            sit_entities.entry(sit.id).or_default().insert(p.entity_id);
            if matches!(p.role, Role::Protagonist) {
                protagonists.insert(p.entity_id);
            }
        }
    }

    // Build adjacency: shared entities between situations
    let sit_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();
    let n = sit_ids.len();
    let mut adj: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let empty_set = HashSet::new();

    for i in 0..n {
        let ents_i = sit_entities.get(&sit_ids[i]).unwrap_or(&empty_set);
        for j in (i + 1)..n {
            let ents_j = sit_entities.get(&sit_ids[j]).unwrap_or(&empty_set);
            let shared = ents_i.intersection(ents_j).count();
            if shared > 0 {
                let weight = shared as f64;
                adj.entry(i).or_default().push((j, weight));
                adj.entry(j).or_default().push((i, weight));
            }
        }
    }

    // Label propagation community detection
    let communities = label_propagation(&adj, n);

    // Identify main plot community: the one containing the most protagonist situations
    let mut community_protagonist_count: HashMap<usize, usize> = HashMap::new();
    for (idx, &comm) in communities.iter().enumerate() {
        let sit_ents = sit_entities.get(&sit_ids[idx]).cloned().unwrap_or_default();
        let has_protag = sit_ents.intersection(&protagonists).count();
        *community_protagonist_count.entry(comm).or_insert(0) += has_protag;
    }

    let main_community = community_protagonist_count
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(&comm, _)| comm)
        .unwrap_or(0);

    // Build subplots from non-main communities
    let mut community_situations: HashMap<usize, Vec<Uuid>> = HashMap::new();
    for (idx, &comm) in communities.iter().enumerate() {
        community_situations
            .entry(comm)
            .or_default()
            .push(sit_ids[idx]);
    }

    let mut subplots = Vec::new();
    for (comm, sit_list) in &community_situations {
        if *comm == main_community {
            continue;
        }
        if sit_list.len() < 2 {
            continue; // Skip single-situation "subplots"
        }

        // Gather characters in this subplot
        let mut chars: HashSet<Uuid> = HashSet::new();
        for &sid in sit_list {
            if let Some(ents) = sit_entities.get(&sid) {
                chars.extend(ents);
            }
        }

        // Chapter range
        let chapters: Vec<usize> = sit_list
            .iter()
            .filter_map(|s| chapter_map.get(s).copied())
            .collect();
        let start = chapters.iter().min().copied().unwrap_or(0);
        let end = chapters.iter().max().copied().unwrap_or(0);

        // Check for convergence: does this subplot share a situation with main plot?
        let main_sits: HashSet<Uuid> = community_situations
            .get(&main_community)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let convergence_point = sit_list
            .iter()
            .filter(|s| {
                // A situation "near" the main plot (shared entities with a main plot situation)
                let ents = sit_entities.get(s).cloned().unwrap_or_default();
                main_sits.iter().any(|ms| {
                    let main_ents = sit_entities.get(ms).cloned().unwrap_or_default();
                    ents.intersection(&main_ents).count() > 0
                })
            })
            .max_by_key(|s| chapter_map.get(s).copied().unwrap_or(0))
            .copied();

        let relation = if convergence_point.is_some() {
            SubplotRelation::Convergence
        } else if chars.intersection(&protagonists).count() > 0 {
            SubplotRelation::Complication
        } else {
            SubplotRelation::Independent
        };

        let subplot_idx = subplots.len() + 1;
        subplots.push(Subplot {
            id: Uuid::now_v7(),
            narrative_id: narrative_id.to_string(),
            label: format!("Subplot {}", subplot_idx),
            situations: sit_list.clone(),
            characters: chars.into_iter().collect(),
            start_chapter: start,
            end_chapter: end,
            relation_to_main: relation,
            convergence_point,
        });
    }

    // Compute density per chapter
    let total_chapters = sorted_sits.len();
    let mut density_per_chapter = vec![0usize; total_chapters];
    for subplot in &subplots {
        for ch in subplot.start_chapter..=subplot.end_chapter.min(total_chapters - 1) {
            density_per_chapter[ch] += 1;
        }
    }

    let average_density = if total_chapters > 0 {
        density_per_chapter.iter().sum::<usize>() as f64 / total_chapters as f64
    } else {
        0.0
    };

    let convergent = subplots
        .iter()
        .filter(|s| matches!(s.relation_to_main, SubplotRelation::Convergence))
        .count();
    let convergence_ratio = if subplots.is_empty() {
        0.0
    } else {
        convergent as f64 / subplots.len() as f64
    };

    let analysis = SubplotAnalysis {
        narrative_id: narrative_id.to_string(),
        subplots,
        density_per_chapter,
        average_density,
        convergence_ratio,
    };

    store_subplot_analysis(hg, &analysis)?;
    Ok(analysis)
}

/// Simple label propagation for community detection.
fn label_propagation(adj: &HashMap<usize, Vec<(usize, f64)>>, n: usize) -> Vec<usize> {
    let mut labels: Vec<usize> = (0..n).collect();
    let max_iter = 20;

    for _ in 0..max_iter {
        let mut changed = false;
        for node in 0..n {
            if let Some(neighbors) = adj.get(&node) {
                if neighbors.is_empty() {
                    continue;
                }
                // Weighted vote from neighbors
                let mut votes: HashMap<usize, f64> = HashMap::new();
                for &(neighbor, weight) in neighbors {
                    *votes.entry(labels[neighbor]).or_insert(0.0) += weight;
                }
                if let Some((&best_label, _)) = votes
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    if labels[node] != best_label {
                        labels[node] = best_label;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    labels
}

// ─── Inference Engine ───────────────────────────────────────

pub struct SubplotEngine;

impl InferenceEngine for SubplotEngine {
    fn job_type(&self) -> InferenceJobType {
        InferenceJobType::SubplotDetection
    }

    fn estimate_cost(
        &self,
        _job: &crate::inference::types::InferenceJob,
        _hg: &Hypergraph,
    ) -> Result<u64> {
        Ok(5000)
    }

    fn execute(
        &self,
        job: &crate::inference::types::InferenceJob,
        hg: &Hypergraph,
    ) -> Result<InferenceResult> {
        let narrative_id = crate::analysis::extract_narrative_id(job)?;

        let analysis = detect_subplots(hg, narrative_id)?;

        Ok(InferenceResult {
            job_id: job.id.clone(),
            job_type: InferenceJobType::SubplotDetection,
            target_id: job.target_id,
            result: serde_json::to_value(&analysis)?,
            confidence: 0.7,
            explanation: Some(format!(
                "{} subplots, avg density {:.1}, convergence ratio {:.0}%",
                analysis.subplots.len(),
                analysis.average_density,
                analysis.convergence_ratio * 100.0
            )),
            status: JobStatus::Completed,
            created_at: job.created_at,
            completed_at: Some(Utc::now()),
        })
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::DateTime;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    fn make_entity(hg: &Hypergraph, name: &str, nid: &str) -> Uuid {
        hg.create_entity(Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": name}),
            beliefs: None,
            embedding: None,
            narrative_id: Some(nid.into()),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        })
        .unwrap()
    }

    fn make_sit(hg: &Hypergraph, nid: &str, hour: i64) -> Uuid {
        let start = DateTime::from_timestamp(1700000000 + hour * 3600, 0).unwrap();
        hg.create_situation(Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(format!("Sit at h{}", hour)),
            description: None,
            temporal: crate::types::AllenInterval {
                start: Some(start),
                end: Some(start + chrono::Duration::hours(1)),
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
            raw_content: vec![ContentBlock::text("content")],
            narrative_level: NarrativeLevel::Scene,
            narrative_id: Some(nid.into()),
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
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
        })
        .unwrap()
    }

    #[test]
    fn test_detect_subplots_multi_thread() {
        let hg = test_hg();
        let nid = "subplot-test";

        // Main plot: hero + villain
        let hero = make_entity(&hg, "Hero", nid);
        let villain = make_entity(&hg, "Villain", nid);
        // Subplot: sidekick + love interest (no villain interaction)
        let sidekick = make_entity(&hg, "Sidekick", nid);
        let love = make_entity(&hg, "Love Interest", nid);

        // Main plot situations (hero + villain)
        for i in 0..5 {
            let sid = make_sit(&hg, nid, i * 24);
            hg.add_participant(Participation {
                entity_id: hero,
                situation_id: sid,
                role: Role::Protagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
            hg.add_participant(Participation {
                entity_id: villain,
                situation_id: sid,
                role: Role::Antagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        // Subplot situations (sidekick + love interest, no hero/villain)
        for i in 5..9 {
            let sid = make_sit(&hg, nid, i * 24);
            hg.add_participant(Participation {
                entity_id: sidekick,
                situation_id: sid,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
            hg.add_participant(Participation {
                entity_id: love,
                situation_id: sid,
                role: Role::Bystander,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let analysis = detect_subplots(&hg, nid).unwrap();
        // Should detect at least 1 subplot (sidekick + love thread)
        assert!(
            !analysis.subplots.is_empty(),
            "Should detect subplot separate from main plot"
        );
    }

    #[test]
    fn test_no_subplots_single_thread() {
        let hg = test_hg();
        let nid = "single-thread";

        let hero = make_entity(&hg, "Hero", nid);
        for i in 0..5 {
            let sid = make_sit(&hg, nid, i * 24);
            hg.add_participant(Participation {
                entity_id: hero,
                situation_id: sid,
                role: Role::Protagonist,
                info_set: None,
                action: None,
                payoff: None,
                seq: 0,
            })
            .unwrap();
        }

        let analysis = detect_subplots(&hg, nid).unwrap();
        assert!(analysis.subplots.is_empty());
        assert_eq!(analysis.convergence_ratio, 0.0);
    }

    #[test]
    fn test_subplot_kv_persistence() {
        let hg = test_hg();
        let analysis = SubplotAnalysis {
            narrative_id: "persist".into(),
            subplots: vec![Subplot {
                id: Uuid::now_v7(),
                narrative_id: "persist".into(),
                label: "Subplot 1".into(),
                situations: vec![Uuid::nil()],
                characters: vec![Uuid::nil()],
                start_chapter: 0,
                end_chapter: 5,
                relation_to_main: SubplotRelation::Convergence,
                convergence_point: Some(Uuid::nil()),
            }],
            density_per_chapter: vec![1, 1, 1, 0, 0, 1],
            average_density: 0.67,
            convergence_ratio: 1.0,
        };
        store_subplot_analysis(&hg, &analysis).unwrap();
        let loaded = load_subplot_analysis(&hg, "persist").unwrap().unwrap();
        assert_eq!(loaded.subplots.len(), 1);
        assert_eq!(
            loaded.subplots[0].relation_to_main,
            SubplotRelation::Convergence
        );
    }

    #[test]
    fn test_empty_narrative() {
        let hg = test_hg();
        let analysis = detect_subplots(&hg, "nonexistent").unwrap();
        assert!(analysis.subplots.is_empty());
    }

    #[test]
    fn test_label_propagation() {
        // Two disconnected cliques
        let mut adj = HashMap::new();
        adj.insert(0, vec![(1, 1.0), (2, 1.0)]);
        adj.insert(1, vec![(0, 1.0), (2, 1.0)]);
        adj.insert(2, vec![(0, 1.0), (1, 1.0)]);
        adj.insert(3, vec![(4, 1.0)]);
        adj.insert(4, vec![(3, 1.0)]);

        let labels = label_propagation(&adj, 5);
        // Nodes 0,1,2 should be same community; 3,4 same community; different from each other
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }
}
