use std::collections::{HashSet, VecDeque};

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::store::TxnOp;
use crate::types::*;
use uuid::Uuid;

impl Hypergraph {
    /// Add a causal link between two situations.
    /// Validates both situations exist and checks for cycles via DFS.
    /// Writes dual index: `c/{from}/{to}` and `cr/{to}/{from}`.
    pub fn add_causal_link(&self, link: CausalLink) -> Result<()> {
        // Validate both situations exist
        self.get_situation(&link.from_situation)?;
        self.get_situation(&link.to_situation)?;

        // Cycle detection: check if to_situation can already reach from_situation
        if self.can_reach(&link.to_situation, &link.from_situation)? {
            return Err(TensaError::CausalCycle {
                from: link.from_situation,
                to: link.to_situation,
            });
        }

        let forward_key = keys::causal_key(&link.from_situation, &link.to_situation);
        let reverse_key = keys::causal_reverse_key(&link.to_situation, &link.from_situation);
        let bytes = serde_json::to_vec(&link)?;

        self.store.transaction(vec![
            TxnOp::Put(forward_key, bytes.clone()),
            TxnOp::Put(reverse_key, bytes),
        ])
    }

    /// Remove a causal link. Deletes both forward and reverse keys.
    pub fn remove_causal_link(&self, from_id: &Uuid, to_id: &Uuid) -> Result<()> {
        let forward_key = keys::causal_key(from_id, to_id);
        if self.store.get(&forward_key)?.is_none() {
            return Err(TensaError::NotFound(format!(
                "Causal link {} -> {}",
                from_id, to_id
            )));
        }

        let reverse_key = keys::causal_reverse_key(to_id, from_id);
        self.store
            .transaction(vec![TxnOp::Delete(forward_key), TxnOp::Delete(reverse_key)])
    }

    /// Get all direct consequences of a situation (one hop forward).
    pub fn get_consequences(&self, situation_id: &Uuid) -> Result<Vec<CausalLink>> {
        let prefix = keys::causal_prefix(situation_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let link: CausalLink = serde_json::from_slice(&value)?;
            result.push(link);
        }
        Ok(result)
    }

    /// Get all direct antecedents of a situation (one hop backward).
    pub fn get_antecedents(&self, situation_id: &Uuid) -> Result<Vec<CausalLink>> {
        let prefix = keys::causal_reverse_prefix(situation_id);
        let pairs = self.store.prefix_scan(&prefix)?;
        let mut result = Vec::new();
        for (_key, value) in pairs {
            let link: CausalLink = serde_json::from_slice(&value)?;
            result.push(link);
        }
        Ok(result)
    }

    /// Traverse the causal chain forward from a situation up to `max_depth` hops.
    /// Returns all reachable CausalLinks in BFS order.
    pub fn traverse_causal_chain(
        &self,
        start_id: &Uuid,
        max_depth: usize,
    ) -> Result<Vec<CausalLink>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(Uuid, usize)> = VecDeque::new();
        queue.push_back((*start_id, 0));
        visited.insert(*start_id);

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let consequences = self.get_consequences(&current)?;
            for link in consequences {
                // Skip soft-deleted or missing target situations
                if self.get_situation(&link.to_situation).is_err() {
                    continue;
                }
                result.push(link.clone());
                if visited.insert(link.to_situation) {
                    queue.push_back((link.to_situation, depth + 1));
                }
            }
        }
        Ok(result)
    }

    /// Check if `from` can reach `target` via forward causal links (DFS).
    fn can_reach(&self, from: &Uuid, target: &Uuid) -> Result<bool> {
        let mut stack = vec![*from];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if current == *target {
                return Ok(true);
            }
            if !visited.insert(current) {
                continue;
            }
            let consequences = self.get_consequences(&current)?;
            for link in consequences {
                stack.push(link.to_situation);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;
    use std::sync::Arc;

    fn test_store() -> Arc<MemoryStore> {
        Arc::new(MemoryStore::new())
    }

    fn setup_situation(hg: &Hypergraph) -> Uuid {
        let sit = Situation {
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
            raw_content: vec![ContentBlock::text("Test")],
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
        };
        hg.create_situation(sit).unwrap()
    }

    fn make_link(from: Uuid, to: Uuid) -> CausalLink {
        CausalLink {
            from_situation: from,
            to_situation: to,
            mechanism: Some("test".to_string()),
            strength: 0.8,
            causal_type: CausalType::Contributing,
            maturity: MaturityLevel::Candidate,
        }
    }

    #[test]
    fn test_add_causal_link() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);
        hg.add_causal_link(make_link(s1, s2)).unwrap();

        let consequences = hg.get_consequences(&s1).unwrap();
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].to_situation, s2);
    }

    #[test]
    fn test_get_consequences() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);
        let s3 = setup_situation(&hg);

        hg.add_causal_link(make_link(s1, s2)).unwrap();
        hg.add_causal_link(make_link(s1, s3)).unwrap();

        let consequences = hg.get_consequences(&s1).unwrap();
        assert_eq!(consequences.len(), 2);
    }

    #[test]
    fn test_get_antecedents() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);
        let s3 = setup_situation(&hg);

        hg.add_causal_link(make_link(s1, s3)).unwrap();
        hg.add_causal_link(make_link(s2, s3)).unwrap();

        let antecedents = hg.get_antecedents(&s3).unwrap();
        assert_eq!(antecedents.len(), 2);
    }

    #[test]
    fn test_causal_chain_forward() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);
        let s3 = setup_situation(&hg);

        hg.add_causal_link(make_link(s1, s2)).unwrap();
        hg.add_causal_link(make_link(s2, s3)).unwrap();

        let chain = hg.traverse_causal_chain(&s1, 10).unwrap();
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_causal_chain_depth_limit() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);
        let s3 = setup_situation(&hg);

        hg.add_causal_link(make_link(s1, s2)).unwrap();
        hg.add_causal_link(make_link(s2, s3)).unwrap();

        // Depth 1: only direct consequences
        let chain = hg.traverse_causal_chain(&s1, 1).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].to_situation, s2);
    }

    #[test]
    fn test_causal_cycle_detection() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);
        let s3 = setup_situation(&hg);

        hg.add_causal_link(make_link(s1, s2)).unwrap();
        hg.add_causal_link(make_link(s2, s3)).unwrap();

        // s3 -> s1 would create a cycle
        let result = hg.add_causal_link(make_link(s3, s1));
        assert!(matches!(result, Err(TensaError::CausalCycle { .. })));
    }

    #[test]
    fn test_causal_link_with_confidence() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);

        let mut link = make_link(s1, s2);
        link.strength = 0.95;
        link.causal_type = CausalType::Necessary;
        hg.add_causal_link(link).unwrap();

        let consequences = hg.get_consequences(&s1).unwrap();
        assert_eq!(consequences[0].strength, 0.95);
        assert_eq!(consequences[0].causal_type, CausalType::Necessary);
    }

    #[test]
    fn test_causal_link_types() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);

        let mut link = make_link(s1, s2);
        link.causal_type = CausalType::Enabling;
        link.mechanism = Some("financial support".to_string());
        hg.add_causal_link(link).unwrap();

        let consequences = hg.get_consequences(&s1).unwrap();
        assert_eq!(consequences[0].causal_type, CausalType::Enabling);
        assert_eq!(
            consequences[0].mechanism.as_deref(),
            Some("financial support")
        );
    }

    #[test]
    fn test_remove_causal_link() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);

        hg.add_causal_link(make_link(s1, s2)).unwrap();
        hg.remove_causal_link(&s1, &s2).unwrap();

        assert!(hg.get_consequences(&s1).unwrap().is_empty());
        assert!(hg.get_antecedents(&s2).unwrap().is_empty());
    }

    #[test]
    fn test_remove_causal_link_not_found() {
        let hg = Hypergraph::new(test_store());
        let result = hg.remove_causal_link(&Uuid::now_v7(), &Uuid::now_v7());
        assert!(matches!(result, Err(TensaError::NotFound(_))));
    }

    #[test]
    fn test_causal_link_maturity() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);
        let s2 = setup_situation(&hg);

        let mut link = make_link(s1, s2);
        link.maturity = MaturityLevel::Validated;
        hg.add_causal_link(link).unwrap();

        let consequences = hg.get_consequences(&s1).unwrap();
        assert_eq!(consequences[0].maturity, MaturityLevel::Validated);
    }

    #[test]
    fn test_causal_self_link_rejected() {
        let hg = Hypergraph::new(test_store());
        let s1 = setup_situation(&hg);

        let result = hg.add_causal_link(make_link(s1, s1));
        assert!(matches!(result, Err(TensaError::CausalCycle { .. })));
    }
}
