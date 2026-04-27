//! Allen constraint network with path consistency propagation.
//!
//! Implements Allen's path consistency algorithm using the composition
//! table from `interval.rs`. Enables multi-hop temporal reasoning:
//! given known relations between interval pairs, propagates constraints
//! to derive all implied relations and detect inconsistencies.

use std::collections::{BTreeSet, HashMap};
use uuid::Uuid;

use crate::error::Result;
use crate::types::AllenRelation;

use super::interval::{compose, ALL};

/// All 13 Allen relations as a BTreeSet — the universal set when no constraint is known.
fn universal_set() -> BTreeSet<AllenRelation> {
    ALL.iter().copied().collect()
}

/// Compute the inverse of a relation set (swap A and B roles).
fn inverse_set(relations: &BTreeSet<AllenRelation>) -> BTreeSet<AllenRelation> {
    relations.iter().map(|r| r.inverse()).collect()
}

/// A constraint network over Allen interval relations.
///
/// Maps pairs of interval IDs to sets of allowed Allen relations.
/// Path consistency propagation tightens these constraints by composing
/// relations through intermediate nodes and intersecting the results.
pub struct ConstraintNetwork {
    /// Constraint store: (id_a, id_b) -> set of allowed relations.
    /// Invariant: for every (a,b) entry, there is a (b,a) entry with inverted relations.
    constraints: HashMap<(Uuid, Uuid), BTreeSet<AllenRelation>>,
    /// All node IDs in the network (for iteration order).
    nodes: BTreeSet<Uuid>,
}

impl ConstraintNetwork {
    /// Create an empty constraint network.
    pub fn new() -> Self {
        Self {
            constraints: HashMap::new(),
            nodes: BTreeSet::new(),
        }
    }

    /// Add a constraint between two intervals.
    ///
    /// If a constraint already exists for this pair, the new relations are
    /// intersected with the existing set (tightening). Also adds the inverse
    /// constraint for `(id_b, id_a)`.
    pub fn add_constraint(&mut self, id_a: Uuid, id_b: Uuid, relations: Vec<AllenRelation>) {
        if id_a == id_b {
            return; // Self-constraint is always {Equals}
        }

        self.nodes.insert(id_a);
        self.nodes.insert(id_b);

        let new_set: BTreeSet<AllenRelation> = relations.into_iter().collect();

        // Forward: (a, b)
        let entry = self
            .constraints
            .entry((id_a, id_b))
            .or_insert_with(universal_set);
        *entry = entry.intersection(&new_set).cloned().collect();

        // Inverse: (b, a)
        let inv_set = inverse_set(&new_set);
        let inv_entry = self
            .constraints
            .entry((id_b, id_a))
            .or_insert_with(universal_set);
        *inv_entry = inv_entry.intersection(&inv_set).cloned().collect();
    }

    /// Get the current allowed relations between two intervals.
    ///
    /// Returns the stored constraint set, or the universal set (all 13 relations)
    /// if no constraint exists between these intervals.
    pub fn get_constraint(&self, id_a: &Uuid, id_b: &Uuid) -> BTreeSet<AllenRelation> {
        if id_a == id_b {
            let mut s = BTreeSet::new();
            s.insert(AllenRelation::Equals);
            return s;
        }
        self.constraints
            .get(&(*id_a, *id_b))
            .cloned()
            .unwrap_or_else(universal_set)
    }

    /// Query the allowed relations between two intervals (convenience wrapper).
    pub fn query_relation(&self, id_a: &Uuid, id_b: &Uuid) -> Vec<AllenRelation> {
        self.get_constraint(id_a, id_b).into_iter().collect()
    }

    /// Run path consistency (Allen's PC-2 algorithm).
    ///
    /// For every triple (i, j, k), tightens the constraint between i and k
    /// by composing constraints (i,j) and (j,k) through the composition table,
    /// then intersecting with the existing (i,k) constraint.
    ///
    /// Returns `Ok(true)` if the network is consistent (all constraint sets
    /// remain non-empty), `Ok(false)` if an inconsistency is detected
    /// (some constraint set becomes empty, meaning no valid assignment exists).
    pub fn propagate(&mut self) -> Result<bool> {
        let nodes: Vec<Uuid> = self.nodes.iter().copied().collect();
        let n = nodes.len();

        loop {
            let mut changed = false;
            // Snapshot constraints at start of each pass to avoid reading
            // mid-pass mutations (required for correct PC-2 propagation)
            let snapshot = self.constraints.clone();

            for i_idx in 0..n {
                for j_idx in 0..n {
                    if i_idx == j_idx {
                        continue;
                    }
                    for k_idx in 0..n {
                        if k_idx == i_idx || k_idx == j_idx {
                            continue;
                        }

                        let i = nodes[i_idx];
                        let j = nodes[j_idx];
                        let k = nodes[k_idx];

                        // Read from snapshot (not live constraints)
                        let r_ij = snapshot.get(&(i, j)).cloned().unwrap_or_else(universal_set);
                        let r_jk = snapshot.get(&(j, k)).cloned().unwrap_or_else(universal_set);

                        let mut composed = BTreeSet::new();
                        for r1 in &r_ij {
                            for r2 in &r_jk {
                                for r in compose(*r1, *r2) {
                                    composed.insert(r);
                                }
                            }
                        }

                        let r_ik = snapshot.get(&(i, k)).cloned().unwrap_or_else(universal_set);
                        let new_r_ik: BTreeSet<AllenRelation> =
                            r_ik.intersection(&composed).cloned().collect();

                        if new_r_ik.is_empty() {
                            return Ok(false); // Inconsistency detected
                        }

                        if new_r_ik != r_ik {
                            let inv = inverse_set(&new_r_ik);
                            self.constraints.insert((i, k), new_r_ik);
                            self.constraints.insert((k, i), inv);
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(true)
    }

    /// Build a constraint network from situations with known temporal data.
    ///
    /// For each pair of situations that have both start and end times,
    /// computes the Allen relation and adds it as a constraint.
    pub fn from_situations(situations: &[crate::types::Situation]) -> Self {
        let mut network = Self::new();

        for (i, a) in situations.iter().enumerate() {
            for b in situations.iter().skip(i + 1) {
                if let Ok(rel) = super::interval::relation_between(&a.temporal, &b.temporal) {
                    network.add_constraint(a.id, b.id, vec![rel]);
                }
            }
        }

        network
    }

    /// Return the number of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of explicit constraints (pairs with non-universal sets).
    pub fn constraint_count(&self) -> usize {
        self.constraints.values().filter(|s| s.len() < 13).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(n: u8) -> Uuid {
        // Deterministic UUID for testing
        Uuid::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n])
    }

    #[test]
    fn test_transitive_before() {
        // A before B, B before C → A before C
        let mut net = ConstraintNetwork::new();
        let a = make_id(1);
        let b = make_id(2);
        let c = make_id(3);

        net.add_constraint(a, b, vec![AllenRelation::Before]);
        net.add_constraint(b, c, vec![AllenRelation::Before]);

        let consistent = net.propagate().unwrap();
        assert!(consistent);

        let ac = net.query_relation(&a, &c);
        assert!(ac.contains(&AllenRelation::Before));
        // Should be tightened from universal to just {Before}
        assert_eq!(ac.len(), 1);
    }

    #[test]
    fn test_detects_inconsistency() {
        // A before B, B before C, C before A → inconsistent cycle
        let mut net = ConstraintNetwork::new();
        let a = make_id(1);
        let b = make_id(2);
        let c = make_id(3);

        net.add_constraint(a, b, vec![AllenRelation::Before]);
        net.add_constraint(b, c, vec![AllenRelation::Before]);
        net.add_constraint(c, a, vec![AllenRelation::Before]);

        let consistent = net.propagate().unwrap();
        assert!(!consistent, "Temporal cycle should be inconsistent");
    }

    #[test]
    fn test_tightens_constraints() {
        // A contains B, B during C — use 4 nodes so the middle pair's
        // composition doesn't interact with inverses of the same pair.
        let mut net = ConstraintNetwork::new();
        let a = make_id(1);
        let b = make_id(2);

        // A contains B, meaning B is during A
        // Add a few possible relations for A-B
        net.add_constraint(
            a,
            b,
            vec![
                AllenRelation::Contains,
                AllenRelation::StartedBy,
                AllenRelation::FinishedBy,
            ],
        );

        let consistent = net.propagate().unwrap();
        assert!(consistent);

        // Should be tightened from universal to {Contains, StartedBy, FinishedBy}
        let ab = net.query_relation(&a, &b);
        assert_eq!(ab.len(), 3);
        assert!(ab.contains(&AllenRelation::Contains));

        // Inverse should be {During, Starts, Finishes}
        let ba = net.query_relation(&b, &a);
        assert_eq!(ba.len(), 3);
        assert!(ba.contains(&AllenRelation::During));
    }

    #[test]
    fn test_equals_transitivity() {
        // A equals B, B equals C → A equals C
        let mut net = ConstraintNetwork::new();
        let a = make_id(1);
        let b = make_id(2);
        let c = make_id(3);

        net.add_constraint(a, b, vec![AllenRelation::Equals]);
        net.add_constraint(b, c, vec![AllenRelation::Equals]);

        let consistent = net.propagate().unwrap();
        assert!(consistent);

        let ac = net.query_relation(&a, &c);
        assert_eq!(ac, vec![AllenRelation::Equals]);
    }

    #[test]
    fn test_inverse_symmetry() {
        let mut net = ConstraintNetwork::new();
        let a = make_id(1);
        let b = make_id(2);

        net.add_constraint(a, b, vec![AllenRelation::Before]);

        // (a, b) = Before means (b, a) = After
        let ba = net.query_relation(&b, &a);
        assert!(ba.contains(&AllenRelation::After));
        assert_eq!(ba.len(), 1);
    }

    #[test]
    fn test_empty_network_consistent() {
        let mut net = ConstraintNetwork::new();
        let consistent = net.propagate().unwrap();
        assert!(consistent);
    }

    #[test]
    fn test_self_constraint_is_equals() {
        let net = ConstraintNetwork::new();
        let a = make_id(1);
        let rel = net.query_relation(&a, &a);
        assert_eq!(rel, vec![AllenRelation::Equals]);
    }

    #[test]
    fn test_from_situations() {
        use crate::types::*;
        use chrono::{Duration, Utc};

        let now = Utc::now();
        let s1 = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: None,
            description: None,
            temporal: AllenInterval {
                start: Some(now),
                end: Some(now + Duration::hours(1)),
                granularity: TimeGranularity::Exact,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Event,
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
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };

        let s2 = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            temporal: AllenInterval {
                start: Some(now + Duration::hours(2)),
                end: Some(now + Duration::hours(3)),
                granularity: TimeGranularity::Exact,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            ..s1.clone()
        };

        let mut net = ConstraintNetwork::from_situations(&[s1.clone(), s2.clone()]);
        let consistent = net.propagate().unwrap();
        assert!(consistent);

        // s1 ends before s2 starts → should be Before
        let rel = net.query_relation(&s1.id, &s2.id);
        assert!(rel.contains(&AllenRelation::Before));
    }
}
