//! Shared graph projection utilities for analysis algorithms.
//!
//! Extracts typed graph projections from the hypergraph:
//! - `CoGraph`: entity co-participation graph (undirected, weighted by shared situations)
//! - `CausalDag`: situation-to-situation causal DAG (directed, weighted by confidence)
//! - `BipartiteGraph`: entity↔situation bipartite graph
//! - `TemporalCoGraph`: time-filtered co-participation graph
//!
//! Also provides Level 0 internal graph algorithms:
//! - WCC (Weakly Connected Components) — BFS flood fill
//! - SCC (Tarjan's Strongly Connected Components)
//! - Topological Sort (Kahn's algorithm)

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

// ─── Data Structures ───────���────────────────────────────────

/// Adjacency list representation of the co-participation graph.
/// Undirected: adj[i] contains (neighbor_idx, shared_situation_count).
pub struct CoGraph {
    /// Entity UUIDs indexed by position.
    pub entities: Vec<Uuid>,
    /// Adjacency: adj[i] = vec of (neighbor_idx, weight).
    pub adj: Vec<Vec<(usize, usize)>>,
}

impl CoGraph {
    /// Get the unweighted neighbor set for a node.
    pub fn neighbor_set(&self, idx: usize) -> HashSet<usize> {
        self.adj[idx].iter().map(|&(v, _)| v).collect()
    }
}

/// Directed acyclic graph of causal links between situations.
/// adj[i] = vec of (successor_idx, causal_confidence).
pub struct CausalDag {
    /// Situation UUIDs indexed by position.
    pub situations: Vec<Uuid>,
    /// Forward adjacency: adj[i] = vec of (to_idx, confidence_weight).
    pub adj: Vec<Vec<(usize, f64)>>,
}

/// Bipartite graph linking entities to situations via participation.
pub struct BipartiteGraph {
    /// Entity UUIDs (left partition).
    pub entities: Vec<Uuid>,
    /// Situation UUIDs (right partition).
    pub situations: Vec<Uuid>,
    /// Edges: (entity_idx, situation_idx).
    pub edges: Vec<(usize, usize)>,
}

// ─── Shared Participation Index ─────────────────────────────

/// Build a situation → [entity_idx] participation index using situation-first scans.
///
/// Instead of N entity-to-situation scans (O(N) prefix scans on `p/`), this
/// does S situation-to-participant scans (via `ps/` reverse index), which
/// directly produces the reverse map needed for co-occurrence counting.
///
/// `situation_filter`: optional set of situation IDs to restrict scanning to.
pub fn collect_participation_index(
    hypergraph: &Hypergraph,
    entity_idx: &HashMap<Uuid, usize>,
    situations: &[Uuid],
    situation_filter: Option<&HashSet<Uuid>>,
) -> Result<HashMap<Uuid, Vec<usize>>> {
    let mut index: HashMap<Uuid, Vec<usize>> = HashMap::new();
    for &sid in situations {
        if let Some(filter) = situation_filter {
            if !filter.contains(&sid) {
                continue;
            }
        }
        let participants = hypergraph.get_participants_for_situation(&sid)?;
        let mut entity_idxs = Vec::new();
        for p in participants {
            if let Some(&idx) = entity_idx.get(&p.entity_id) {
                entity_idxs.push(idx);
            }
        }
        if !entity_idxs.is_empty() {
            index.insert(sid, entity_idxs);
        }
    }
    Ok(index)
}

// ─── Graph Extraction ──────────────────────────────────────

/// Build the co-participation graph from the hypergraph for a given narrative.
///
/// Two entities share an edge if they co-participate in at least one situation.
/// Edge weight = number of shared situations.
pub fn build_co_graph(hypergraph: &Hypergraph, narrative_id: &str) -> Result<CoGraph> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();

    if entity_ids.is_empty() {
        return Ok(CoGraph {
            entities: vec![],
            adj: vec![],
        });
    }

    let n = entity_ids.len();
    let entity_idx: HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let sit_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();

    // Situation-first scan: S scans via ps/ reverse index instead of N entity scans
    let situation_entities = collect_participation_index(hypergraph, &entity_idx, &sit_ids, None)?;

    // Count co-occurrences.
    let mut edge_weights: HashMap<(usize, usize), usize> = HashMap::new();
    for (_sid, ents) in &situation_entities {
        for a in 0..ents.len() {
            for b in (a + 1)..ents.len() {
                let (lo, hi) = if ents[a] < ents[b] {
                    (ents[a], ents[b])
                } else {
                    (ents[b], ents[a])
                };
                *edge_weights.entry((lo, hi)).or_insert(0) += 1;
            }
        }
    }

    // Build adjacency list.
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
    for (&(a, b), &w) in &edge_weights {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }

    Ok(CoGraph {
        entities: entity_ids,
        adj,
    })
}

/// Build the causal DAG from situation→situation causal links.
///
/// Returns a directed graph where edge weight = causal link confidence.
pub fn build_causal_dag(hypergraph: &Hypergraph, narrative_id: &str) -> Result<CausalDag> {
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let sit_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();

    if sit_ids.is_empty() {
        return Ok(CausalDag {
            situations: vec![],
            adj: vec![],
        });
    }

    let n = sit_ids.len();
    let id_to_idx: HashMap<Uuid, usize> =
        sit_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

    for (i, sid) in sit_ids.iter().enumerate() {
        let consequences = hypergraph.get_consequences(sid)?;
        for link in consequences {
            if let Some(&j) = id_to_idx.get(&link.to_situation) {
                adj[i].push((j, link.strength as f64));
            }
        }
    }

    Ok(CausalDag {
        situations: sit_ids,
        adj,
    })
}

/// Build an entity-level graph induced by the situation-level causal DAG.
///
/// For every causal link `s₁ → s₂`, connect every participant of `s₁` to every
/// participant of `s₂`. Edge weight = number of distinct causal links joining
/// the two entities' situations. Undirected: passing through causal chains
/// either way counts as a bridge.
///
/// Use this projection when the co-participation graph is too dense to
/// surface cut-vertices — typical of post-event / post-arrest narratives
/// where many actors share a single "raid" situation and trivially edge
/// each other. The causal projection only connects actors whose actions
/// actually feed into each other's outcomes.
pub fn build_causal_entity_graph(hypergraph: &Hypergraph, narrative_id: &str) -> Result<CoGraph> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    if entity_ids.is_empty() {
        return Ok(CoGraph {
            entities: vec![],
            adj: vec![],
        });
    }
    let n = entity_ids.len();
    let entity_idx: HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let sit_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();
    let sit_participants = collect_participation_index(hypergraph, &entity_idx, &sit_ids, None)?;

    let mut edge_weights: HashMap<(usize, usize), usize> = HashMap::new();
    for sid in &sit_ids {
        let from_ents = match sit_participants.get(sid) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        for link in hypergraph.get_consequences(sid)? {
            let Some(to_ents) = sit_participants.get(&link.to_situation) else {
                continue;
            };
            for &a in from_ents {
                for &b in to_ents {
                    if a == b {
                        continue;
                    }
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    *edge_weights.entry((lo, hi)).or_insert(0) += 1;
                }
            }
        }
    }

    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
    for (&(a, b), &w) in &edge_weights {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    Ok(CoGraph {
        entities: entity_ids,
        adj,
    })
}

/// Build the entity↔situation bipartite graph for a narrative.
pub fn build_bipartite(hypergraph: &Hypergraph, narrative_id: &str) -> Result<BipartiteGraph> {
    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();
    let situation_ids: Vec<Uuid> = situations.iter().map(|s| s.id).collect();

    let entity_idx: HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let sit_id_to_idx: HashMap<Uuid, usize> = situation_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Situation-first scan via shared helper
    let sit_entities = collect_participation_index(hypergraph, &entity_idx, &situation_ids, None)?;

    let mut edges = Vec::new();
    for (&sid, ent_idxs) in &sit_entities {
        if let Some(&si) = sit_id_to_idx.get(&sid) {
            for &ei in ent_idxs {
                edges.push((ei, si));
            }
        }
    }

    Ok(BipartiteGraph {
        entities: entity_ids,
        situations: situation_ids,
        edges,
    })
}

/// Build a time-filtered co-participation graph.
///
/// Only includes co-occurrences in situations whose temporal interval
/// overlaps with the given time window. If `window` is None, behaves
/// like `build_co_graph`.
pub fn build_temporal_graph(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    window: Option<(DateTime<Utc>, DateTime<Utc>)>,
) -> Result<CoGraph> {
    if window.is_none() {
        return build_co_graph(hypergraph, narrative_id);
    }
    let (win_start, win_end) = window.unwrap();

    let entities = hypergraph.list_entities_by_narrative(narrative_id)?;
    let entity_ids: Vec<Uuid> = entities.iter().map(|e| e.id).collect();

    if entity_ids.is_empty() {
        return Ok(CoGraph {
            entities: vec![],
            adj: vec![],
        });
    }

    let n = entity_ids.len();
    let entity_idx: HashMap<Uuid, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Collect situations that overlap with the window.
    let all_situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    let valid_situations: HashSet<Uuid> = all_situations
        .iter()
        .filter(|s| {
            let s_start = s.temporal.start.unwrap_or(DateTime::<Utc>::MIN_UTC);
            let s_end = s.temporal.end.unwrap_or(DateTime::<Utc>::MAX_UTC);
            s_start <= win_end && s_end >= win_start
        })
        .map(|s| s.id)
        .collect();

    let valid_sit_ids: Vec<Uuid> = valid_situations.iter().copied().collect();

    // Only scan situations within the time window
    let situation_entities =
        collect_participation_index(hypergraph, &entity_idx, &valid_sit_ids, None)?;

    let mut edge_weights: HashMap<(usize, usize), usize> = HashMap::new();
    for (_sid, ents) in &situation_entities {
        for a in 0..ents.len() {
            for b in (a + 1)..ents.len() {
                let (lo, hi) = if ents[a] < ents[b] {
                    (ents[a], ents[b])
                } else {
                    (ents[b], ents[a])
                };
                *edge_weights.entry((lo, hi)).or_insert(0) += 1;
            }
        }
    }

    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
    for (&(a, b), &w) in &edge_weights {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }

    Ok(CoGraph {
        entities: entity_ids,
        adj,
    })
}

// ─── Level 0: Weakly Connected Components ───────────────────

/// Compute weakly connected components of an undirected co-graph via BFS.
///
/// Returns a list of components, each a Vec of node indices.
/// Used internally by closeness (Wasserman-Faust correction) and Leiden seeding.
pub fn wcc(graph: &CoGraph) -> Vec<Vec<usize>> {
    let n = graph.entities.len();
    if n == 0 {
        return vec![];
    }

    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for &(neighbor, _) in &graph.adj[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        components.push(component);
    }

    components
}

// ─── Level 0: Strongly Connected Components (Tarjan) ────────

/// Compute strongly connected components of a CausalDag using iterative Tarjan's algorithm.
///
/// Returns a list of components, each a Vec of node indices.
/// Uses an explicit call stack to avoid stack overflow on large graphs.
/// In a proper DAG, every SCC should be a singleton — multiple-node SCCs
/// indicate cycles (which should have been prevented by add_causal_link).
pub fn scc(dag: &CausalDag) -> Vec<Vec<usize>> {
    let n = dag.situations.len();
    if n == 0 {
        return vec![];
    }

    let mut index_counter: usize = 0;
    let mut tarjan_stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices: Vec<Option<usize>> = vec![None; n];
    let mut lowlinks = vec![0usize; n];
    let mut components: Vec<Vec<usize>> = Vec::new();

    // Explicit call stack: (node, edge_cursor)
    // edge_cursor tracks which neighbor we're about to visit next
    let mut call_stack: Vec<(usize, usize)> = Vec::new();

    for root in 0..n {
        if indices[root].is_some() {
            continue;
        }

        // Initialize root
        indices[root] = Some(index_counter);
        lowlinks[root] = index_counter;
        index_counter += 1;
        tarjan_stack.push(root);
        on_stack[root] = true;
        call_stack.push((root, 0));

        while let Some(&mut (v, ref mut cursor)) = call_stack.last_mut() {
            if *cursor < dag.adj[v].len() {
                let (w, _) = dag.adj[v][*cursor];
                *cursor += 1;

                if indices[w].is_none() {
                    // Tree edge: push w onto call stack (simulates recursion)
                    indices[w] = Some(index_counter);
                    lowlinks[w] = index_counter;
                    index_counter += 1;
                    tarjan_stack.push(w);
                    on_stack[w] = true;
                    call_stack.push((w, 0));
                } else if on_stack[w] {
                    lowlinks[v] = lowlinks[v].min(indices[w].unwrap());
                }
            } else {
                // All neighbors visited — check if v is an SCC root
                if lowlinks[v] == indices[v].unwrap() {
                    let mut component = Vec::new();
                    loop {
                        let w = tarjan_stack.pop().unwrap();
                        on_stack[w] = false;
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    components.push(component);
                }

                // Pop v and propagate lowlink to parent
                call_stack.pop();
                if let Some(&mut (parent, _)) = call_stack.last_mut() {
                    lowlinks[parent] = lowlinks[parent].min(lowlinks[v]);
                }
            }
        }
    }

    components
}

// ─── Level 0: Topological Sort (Kahn's) ���────────────────────

/// Topological sort of a CausalDag using Kahn's algorithm.
///
/// Returns `Ok(sorted_indices)` if the graph is a DAG, or `Err` if cycles exist.
/// Used by narrative diameter (longest path via DP) and temporal ILP.
pub fn topological_sort(dag: &CausalDag) -> Result<Vec<usize>> {
    let n = dag.situations.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let mut in_degree = vec![0usize; n];
    for neighbors in &dag.adj {
        for &(to, _) in neighbors {
            in_degree[to] += 1;
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut sorted = Vec::with_capacity(n);
    while let Some(node) = queue.pop_front() {
        sorted.push(node);
        for &(to, _) in &dag.adj[node] {
            in_degree[to] -= 1;
            if in_degree[to] == 0 {
                queue.push_back(to);
            }
        }
    }

    if sorted.len() != n {
        return Err(TensaError::InferenceError(
            "Causal graph contains cycles — topological sort impossible".into(),
        ));
    }

    Ok(sorted)
}

// ─── Tests ───────────────��──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::*;
    use crate::types::*;
    use chrono::Duration;

    // ── CoGraph tests ──

    #[test]
    fn test_build_co_graph_basic() {
        let hg = make_hg();
        let nid = "test-cograph";
        let a = add_entity(&hg, "Alice", nid);
        let b = add_entity(&hg, "Bob", nid);
        let c = add_entity(&hg, "Carol", nid);
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);

        link(&hg, a, s1);
        link(&hg, b, s1);
        link(&hg, a, s2);
        link(&hg, c, s2);

        let graph = build_co_graph(&hg, nid).unwrap();
        assert_eq!(graph.entities.len(), 3);
        // a-b share s1, a-c share s2
        let total_edges: usize = graph.adj.iter().map(|a| a.len()).sum();
        assert_eq!(total_edges, 4); // 2 edges × 2 directions
    }

    #[test]
    fn test_build_co_graph_empty() {
        let hg = make_hg();
        let graph = build_co_graph(&hg, "nonexistent").unwrap();
        assert!(graph.entities.is_empty());
        assert!(graph.adj.is_empty());
    }

    #[test]
    fn test_build_causal_dag() {
        let hg = make_hg();
        let nid = "test-dag";
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);

        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            strength: 0.8,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let dag = build_causal_dag(&hg, nid).unwrap();
        assert_eq!(dag.situations.len(), 3);
        // s1→s2, s2→s3
        let total_edges: usize = dag.adj.iter().map(|a| a.len()).sum();
        assert_eq!(total_edges, 2);
    }

    #[test]
    fn test_build_causal_entity_graph_connects_across_causal_links() {
        let hg = make_hg();
        let nid = "test-causal-entity";
        let a = add_entity(&hg, "Alice", nid);
        let b = add_entity(&hg, "Bob", nid);
        let c = add_entity(&hg, "Carol", nid);
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);

        // Alice in s1, Bob in s2 and s3, Carol in s3.
        link(&hg, a, s1);
        link(&hg, b, s2);
        link(&hg, b, s3);
        link(&hg, c, s3);

        // Causal: s1 → s2, s2 → s3.
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            strength: 0.8,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let graph = build_causal_entity_graph(&hg, nid).unwrap();
        assert_eq!(graph.entities.len(), 3);

        // Expect edges A-B (via s1→s2) and B-C (via s2→s3) — but NOT A-C
        // (no causal path between s1 and s3 without Bob's bridging situation).
        let idx = |e| graph.entities.iter().position(|x| *x == e).unwrap();
        let neighbors = |i: usize| -> Vec<usize> { graph.adj[i].iter().map(|&(n, _)| n).collect() };
        assert!(neighbors(idx(a)).contains(&idx(b)));
        assert!(neighbors(idx(b)).contains(&idx(c)));
        assert!(!neighbors(idx(a)).contains(&idx(c)));
    }

    #[test]
    fn test_build_bipartite() {
        let hg = make_hg();
        let nid = "test-bip";
        let a = add_entity(&hg, "Alice", nid);
        let b = add_entity(&hg, "Bob", nid);
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);

        link(&hg, a, s1);
        link(&hg, a, s2);
        link(&hg, b, s1);

        let bip = build_bipartite(&hg, nid).unwrap();
        assert_eq!(bip.entities.len(), 2);
        assert_eq!(bip.situations.len(), 2);
        assert_eq!(bip.edges.len(), 3);
    }

    #[test]
    fn test_build_temporal_graph_filters() {
        let hg = make_hg();
        let nid = "test-temporal";
        let a = add_entity(&hg, "Alice", nid);
        let b = add_entity(&hg, "Bob", nid);

        let base = Utc::now();
        // s1: in the past
        let s1 = {
            let sit = crate::types::Situation {
                id: uuid::Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(base - Duration::hours(10)),
                    end: Some(base - Duration::hours(9)),
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
                raw_content: vec![ContentBlock::text("past")],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.8,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some(nid.to_string()),
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
            };
            hg.create_situation(sit).unwrap()
        };
        // s2: in the future window
        let s2 = {
            let sit = crate::types::Situation {
                id: uuid::Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(base + Duration::hours(1)),
                    end: Some(base + Duration::hours(2)),
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
                raw_content: vec![ContentBlock::text("future")],
                narrative_level: NarrativeLevel::Scene,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.8,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some(nid.to_string()),
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
            };
            hg.create_situation(sit).unwrap()
        };

        link(&hg, a, s1);
        link(&hg, b, s1);
        link(&hg, a, s2);
        link(&hg, b, s2);

        // Window that only includes s2
        let graph =
            build_temporal_graph(&hg, nid, Some((base, base + Duration::hours(3)))).unwrap();
        // Both entities should be linked via s2 only
        let total_edges: usize = graph.adj.iter().map(|a| a.len()).sum();
        assert_eq!(total_edges, 2); // 1 edge × 2 directions
    }

    #[test]
    fn test_empty_narrative() {
        let hg = make_hg();
        let graph = build_co_graph(&hg, "empty").unwrap();
        assert!(graph.entities.is_empty());
        let dag = build_causal_dag(&hg, "empty").unwrap();
        assert!(dag.situations.is_empty());
        let bip = build_bipartite(&hg, "empty").unwrap();
        assert!(bip.entities.is_empty());
    }

    // ── WCC tests ──

    #[test]
    fn test_wcc_single_component() {
        let hg = make_hg();
        let nid = "wcc-single";
        let a = add_entity(&hg, "A", nid);
        let b = add_entity(&hg, "B", nid);
        let c = add_entity(&hg, "C", nid);
        let s = add_situation(&hg, nid);
        link(&hg, a, s);
        link(&hg, b, s);
        link(&hg, c, s);

        let graph = build_co_graph(&hg, nid).unwrap();
        let components = wcc(&graph);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn test_wcc_multiple_components() {
        let hg = make_hg();
        let nid = "wcc-multi";
        let a = add_entity(&hg, "A", nid);
        let b = add_entity(&hg, "B", nid);
        let c = add_entity(&hg, "C", nid);
        let d = add_entity(&hg, "D", nid);
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);

        // a-b in s1, c-d in s2 — two components
        link(&hg, a, s1);
        link(&hg, b, s1);
        link(&hg, c, s2);
        link(&hg, d, s2);

        let graph = build_co_graph(&hg, nid).unwrap();
        let components = wcc(&graph);
        assert_eq!(components.len(), 2);
    }

    // ── SCC tests ──

    #[test]
    fn test_scc_dag_singletons() {
        let hg = make_hg();
        let nid = "scc-dag";
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);

        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            strength: 0.8,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let dag = build_causal_dag(&hg, nid).unwrap();
        let components = scc(&dag);
        // All singletons in a proper DAG
        assert_eq!(components.len(), 3);
        for c in &components {
            assert_eq!(c.len(), 1);
        }
    }

    #[test]
    fn test_scc_with_cycle() {
        // Manually construct a CausalDag with a cycle (bypassing add_causal_link validation)
        let dag = CausalDag {
            situations: vec![Uuid::nil(); 3],
            adj: vec![
                vec![(1, 1.0)], // 0 → 1
                vec![(2, 1.0)], // 1 → 2
                vec![(0, 1.0)], // 2 → 0 (cycle!)
            ],
        };
        let components = scc(&dag);
        // All 3 nodes should be in one SCC
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    // ── Topological Sort tests ──

    #[test]
    fn test_topological_sort_linear() {
        let hg = make_hg();
        let nid = "topo-linear";
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);

        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s3,
            strength: 0.8,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let dag = build_causal_dag(&hg, nid).unwrap();
        let sorted = topological_sort(&dag).unwrap();
        assert_eq!(sorted.len(), 3);

        // Build index map to verify ordering
        let id_to_idx: HashMap<Uuid, usize> = dag
            .situations
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let pos: HashMap<usize, usize> = sorted
            .iter()
            .enumerate()
            .map(|(p, &idx)| (idx, p))
            .collect();

        // s1 must come before s2, s2 before s3
        assert!(pos[&id_to_idx[&s1]] < pos[&id_to_idx[&s2]]);
        assert!(pos[&id_to_idx[&s2]] < pos[&id_to_idx[&s3]]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        let hg = make_hg();
        let nid = "topo-diamond";
        let s1 = add_situation(&hg, nid);
        let s2 = add_situation(&hg, nid);
        let s3 = add_situation(&hg, nid);
        let s4 = add_situation(&hg, nid);

        // Diamond: s1→s2, s1→s3, s2→s4, s3→s4
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s2,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s1,
            to_situation: s3,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s2,
            to_situation: s4,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();
        hg.add_causal_link(CausalLink {
            from_situation: s3,
            to_situation: s4,
            strength: 0.9,
            mechanism: None,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        })
        .unwrap();

        let dag = build_causal_dag(&hg, nid).unwrap();
        let sorted = topological_sort(&dag).unwrap();
        assert_eq!(sorted.len(), 4);

        let id_to_idx: HashMap<Uuid, usize> = dag
            .situations
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let pos: HashMap<usize, usize> = sorted
            .iter()
            .enumerate()
            .map(|(p, &idx)| (idx, p))
            .collect();

        // s1 before s2 and s3; s2 and s3 before s4
        assert!(pos[&id_to_idx[&s1]] < pos[&id_to_idx[&s2]]);
        assert!(pos[&id_to_idx[&s1]] < pos[&id_to_idx[&s3]]);
        assert!(pos[&id_to_idx[&s2]] < pos[&id_to_idx[&s4]]);
        assert!(pos[&id_to_idx[&s3]] < pos[&id_to_idx[&s4]]);
    }

    #[test]
    fn test_topological_sort_cycle_error() {
        // Manually constructed cycle
        let dag = CausalDag {
            situations: vec![Uuid::nil(); 3],
            adj: vec![
                vec![(1, 1.0)],
                vec![(2, 1.0)],
                vec![(0, 1.0)], // cycle
            ],
        };
        assert!(topological_sort(&dag).is_err());
    }
}
