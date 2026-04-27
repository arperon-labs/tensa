//! LoRA adapter training and merging infrastructure (Sprint D9.5).
//!
//! Manages LoRA adapters for author-specific text generation.
//! Training is handled by separate binaries; this module provides
//! adapter metadata management and arithmetic merging.
//!
//! KV persistence at `lora/` prefix.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;

// ─── Types ──────────────────────────────────────────────────

/// LoRA training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainingConfig {
    /// LoRA rank (8–64).
    pub rank: usize,
    /// LoRA alpha (scaling factor).
    pub alpha: f64,
    /// Target modules (e.g., ["q_proj", "v_proj"]).
    pub target_modules: Vec<String>,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Whether to use 4-bit quantization (QLoRA).
    pub use_qlora: bool,
}

impl Default for LoraTrainingConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
            learning_rate: 2e-4,
            epochs: 3,
            batch_size: 4,
            use_qlora: true,
        }
    }
}

/// Training status of a LoRA adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoraStatus {
    /// Training job queued.
    Queued,
    /// Currently training.
    Training,
    /// Training complete, adapter available.
    Ready,
    /// Training failed.
    Failed { error: String },
}

/// Merging strategy for combining multiple LoRAs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Simple weighted average of LoRA weights.
    Linear,
    /// TIES merging: Trim, Elect Sign, Disjoint Merge.
    Ties,
    /// DARE: Drop And Rescale.
    Dare,
}

/// A LoRA adapter for author-specific generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    pub id: Uuid,
    /// Base LLM model this adapter was trained on.
    pub base_model: String,
    /// Author entity ID (if author-specific).
    pub author_id: Option<Uuid>,
    /// Label for identification.
    pub label: String,
    /// Training configuration used.
    pub training_config: LoraTrainingConfig,
    /// Training corpus size (words).
    pub training_corpus_size: usize,
    /// Current status.
    pub status: LoraStatus,
    /// Filesystem path to adapter weights (safetensors).
    pub weights_path: Option<String>,
    /// Training loss curve (per-epoch).
    pub loss_history: Vec<f64>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A merged LoRA adapter combining multiple source adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedLoraAdapter {
    pub id: Uuid,
    pub label: String,
    /// Source adapters and their weights.
    pub sources: Vec<(Uuid, f64)>,
    pub merge_strategy: MergeStrategy,
    pub base_model: String,
    pub weights_path: Option<String>,
    pub status: LoraStatus,
    pub created_at: DateTime<Utc>,
}

// ─── KV Operations ──────────────────────────────────────────

fn lora_key(id: &Uuid) -> Vec<u8> {
    format!("lora/{}", id).into_bytes()
}

fn merged_key(id: &Uuid) -> Vec<u8> {
    format!("lora/merged/{}", id).into_bytes()
}

pub fn store_adapter(hg: &Hypergraph, adapter: &LoraAdapter) -> Result<()> {
    let key = lora_key(&adapter.id);
    let val = serde_json::to_vec(adapter)?;
    hg.store().put(&key, &val)
}

pub fn load_adapter(hg: &Hypergraph, id: &Uuid) -> Result<Option<LoraAdapter>> {
    let key = lora_key(id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

pub fn list_adapters(hg: &Hypergraph) -> Result<Vec<LoraAdapter>> {
    let prefix = b"lora/";
    let items = hg.store().prefix_scan(prefix)?;
    let mut out = Vec::new();
    for (k, v) in items {
        let key_str = String::from_utf8_lossy(&k);
        if key_str.contains("/merged/") {
            continue;
        }
        if let Ok(adapter) = serde_json::from_slice::<LoraAdapter>(&v) {
            out.push(adapter);
        }
    }
    Ok(out)
}

pub fn delete_adapter(hg: &Hypergraph, id: &Uuid) -> Result<()> {
    let key = lora_key(id);
    hg.store().delete(&key)
}

pub fn store_merged(hg: &Hypergraph, merged: &MergedLoraAdapter) -> Result<()> {
    let key = merged_key(&merged.id);
    let val = serde_json::to_vec(merged)?;
    hg.store().put(&key, &val)
}

pub fn load_merged(hg: &Hypergraph, id: &Uuid) -> Result<Option<MergedLoraAdapter>> {
    let key = merged_key(id);
    match hg.store().get(&key)? {
        Some(v) => Ok(Some(serde_json::from_slice(&v)?)),
        None => Ok(None),
    }
}

/// Queue a LoRA training job. Returns the adapter ID.
///
/// The actual training is handled by external infrastructure
/// (e.g., `bin/train_author_lora.rs`). This records the adapter metadata
/// and sets status to Queued.
pub fn queue_training(
    hg: &Hypergraph,
    base_model: &str,
    author_id: Option<Uuid>,
    label: &str,
    config: LoraTrainingConfig,
    corpus_size: usize,
) -> Result<Uuid> {
    let adapter = LoraAdapter {
        id: Uuid::now_v7(),
        base_model: base_model.to_string(),
        author_id,
        label: label.to_string(),
        training_config: config,
        training_corpus_size: corpus_size,
        status: LoraStatus::Queued,
        weights_path: None,
        loss_history: Vec::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    let id = adapter.id;
    store_adapter(hg, &adapter)?;
    Ok(id)
}

/// Request merging of multiple LoRA adapters with given weights.
pub fn queue_merge(
    hg: &Hypergraph,
    sources: Vec<(Uuid, f64)>,
    strategy: MergeStrategy,
    label: &str,
) -> Result<Uuid> {
    // Validate all source adapters exist and are Ready
    let mut base_model = String::new();
    for (id, _) in &sources {
        let adapter = load_adapter(hg, id)?
            .ok_or_else(|| TensaError::NotFound(format!("LoRA adapter {}", id)))?;
        if adapter.status != LoraStatus::Ready {
            return Err(TensaError::InvalidQuery(format!(
                "adapter {} is not ready (status: {:?})",
                id, adapter.status
            )));
        }
        if base_model.is_empty() {
            base_model = adapter.base_model.clone();
        } else if adapter.base_model != base_model {
            return Err(TensaError::InvalidQuery(
                "cannot merge adapters from different base models".into(),
            ));
        }
    }

    let merged = MergedLoraAdapter {
        id: Uuid::now_v7(),
        label: label.to_string(),
        sources,
        merge_strategy: strategy,
        base_model,
        weights_path: None,
        status: LoraStatus::Queued,
        created_at: Utc::now(),
    };
    let id = merged.id;
    store_merged(hg, &merged)?;
    Ok(id)
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn test_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn test_adapter_crud() {
        let hg = test_hg();
        let adapter = LoraAdapter {
            id: Uuid::now_v7(),
            base_model: "Qwen3-14B".into(),
            author_id: Some(Uuid::now_v7()),
            label: "hemingway-lora".into(),
            training_config: LoraTrainingConfig::default(),
            training_corpus_size: 500_000,
            status: LoraStatus::Ready,
            weights_path: Some("/models/hemingway.safetensors".into()),
            loss_history: vec![2.5, 1.8, 1.2],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        store_adapter(&hg, &adapter).unwrap();
        let loaded = load_adapter(&hg, &adapter.id).unwrap().unwrap();
        assert_eq!(loaded.label, "hemingway-lora");
        assert_eq!(loaded.status, LoraStatus::Ready);

        delete_adapter(&hg, &adapter.id).unwrap();
        assert!(load_adapter(&hg, &adapter.id).unwrap().is_none());
    }

    #[test]
    fn test_queue_training() {
        let hg = test_hg();
        let id = queue_training(
            &hg,
            "Qwen3-14B",
            None,
            "test-lora",
            LoraTrainingConfig::default(),
            100_000,
        )
        .unwrap();

        let adapter = load_adapter(&hg, &id).unwrap().unwrap();
        assert_eq!(adapter.status, LoraStatus::Queued);
        assert_eq!(adapter.training_corpus_size, 100_000);
    }

    #[test]
    fn test_queue_merge() {
        let hg = test_hg();

        // Create two ready adapters
        let a1 = LoraAdapter {
            id: Uuid::now_v7(),
            base_model: "base".into(),
            author_id: None,
            label: "a1".into(),
            training_config: LoraTrainingConfig::default(),
            training_corpus_size: 1000,
            status: LoraStatus::Ready,
            weights_path: None,
            loss_history: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let a2 = LoraAdapter {
            id: Uuid::now_v7(),
            base_model: "base".into(),
            author_id: None,
            label: "a2".into(),
            training_config: LoraTrainingConfig::default(),
            training_corpus_size: 2000,
            status: LoraStatus::Ready,
            weights_path: None,
            loss_history: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        store_adapter(&hg, &a1).unwrap();
        store_adapter(&hg, &a2).unwrap();

        let merged_id = queue_merge(
            &hg,
            vec![(a1.id, 0.6), (a2.id, 0.4)],
            MergeStrategy::Linear,
            "blended",
        )
        .unwrap();

        let merged = load_merged(&hg, &merged_id).unwrap().unwrap();
        assert_eq!(merged.sources.len(), 2);
        assert_eq!(merged.merge_strategy, MergeStrategy::Linear);
    }

    #[test]
    fn test_merge_rejects_non_ready() {
        let hg = test_hg();
        let a = LoraAdapter {
            id: Uuid::now_v7(),
            base_model: "base".into(),
            author_id: None,
            label: "queued".into(),
            training_config: LoraTrainingConfig::default(),
            training_corpus_size: 1000,
            status: LoraStatus::Queued,
            weights_path: None,
            loss_history: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        store_adapter(&hg, &a).unwrap();

        let result = queue_merge(&hg, vec![(a.id, 1.0)], MergeStrategy::Linear, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_rejects_different_base_models() {
        let hg = test_hg();
        let a1 = LoraAdapter {
            id: Uuid::now_v7(),
            base_model: "model-a".into(),
            author_id: None,
            label: "a1".into(),
            training_config: LoraTrainingConfig::default(),
            training_corpus_size: 1000,
            status: LoraStatus::Ready,
            weights_path: None,
            loss_history: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let a2 = LoraAdapter {
            id: Uuid::now_v7(),
            base_model: "model-b".into(),
            author_id: None,
            label: "a2".into(),
            training_config: LoraTrainingConfig::default(),
            training_corpus_size: 1000,
            status: LoraStatus::Ready,
            weights_path: None,
            loss_history: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        store_adapter(&hg, &a1).unwrap();
        store_adapter(&hg, &a2).unwrap();

        let result = queue_merge(
            &hg,
            vec![(a1.id, 0.5), (a2.id, 0.5)],
            MergeStrategy::Linear,
            "test",
        );
        assert!(result.is_err());
    }
}
