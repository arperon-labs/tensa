//! Async worker pool for inference job execution.
//!
//! Spawns configurable number of tokio tasks that poll the job queue,
//! dispatch to registered inference engines, and store results.
//! Uses a Semaphore for concurrency control and Notify for wakeup.
//!
//! ## Engine lookup keying
//!
//! `engines` is keyed by [`std::mem::Discriminant<InferenceJobType>`] —
//! NOT by the full enum value. The reason: variants like
//! `InferenceJobType::SurrogateGeneration { source_narrative_id, ... }`
//! carry per-job payload (string IDs, opaque JSON params), and engine
//! registration uses a sentinel value with empty fields. Equality-based
//! lookup would never match a real job. Discriminant-based lookup matches
//! by variant tag, which is exactly the right granularity: one engine
//! handles one variant, regardless of payload.
//!
//! Phase 0 of the EATH sprint flagged this as a deferral; Phase 4 (this
//! commit) closes it. See [`crate::synth::engines`] for the engines that
//! drove the change.

use std::collections::HashMap;
use std::mem::Discriminant;
use std::sync::Arc;

use tokio::sync::{Notify, Semaphore};

use crate::hypergraph::Hypergraph;
use crate::types::InferenceJobType;

use super::jobs::JobQueue;
use super::InferenceEngine;

/// Convenience: turn a job type into the lookup key. Centralized so the
/// "key by discriminant, not by value" decision lives in one place.
#[inline]
fn engine_key(job_type: &InferenceJobType) -> Discriminant<InferenceJobType> {
    std::mem::discriminant(job_type)
}

/// Async worker pool that processes inference jobs.
pub struct WorkerPool {
    job_queue: Arc<JobQueue>,
    hypergraph: Arc<Hypergraph>,
    engines: HashMap<Discriminant<InferenceJobType>, Arc<dyn InferenceEngine>>,
    concurrency: usize,
    notify: Arc<Notify>,
    shutdown: tokio::sync::watch::Sender<bool>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
}

impl WorkerPool {
    /// Create a new worker pool.
    pub fn new(job_queue: Arc<JobQueue>, hypergraph: Arc<Hypergraph>, concurrency: usize) -> Self {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        Self {
            job_queue,
            hypergraph,
            engines: HashMap::new(),
            concurrency,
            notify: Arc::new(Notify::new()),
            shutdown: shutdown_tx,
            shutdown_rx,
        }
    }

    /// Register an inference engine for a job type.
    ///
    /// The engine's [`InferenceEngine::job_type`] is consulted ONLY for its
    /// discriminant — payload values inside variants like
    /// `SurrogateGeneration { ... }` are ignored at registration time. See
    /// the module doc for the rationale.
    pub fn register_engine(&mut self, engine: Arc<dyn InferenceEngine>) {
        let key = engine_key(&engine.job_type());
        self.engines.insert(key, engine);
    }

    /// Get a notifier to wake workers when new jobs arrive.
    pub fn notifier(&self) -> Arc<Notify> {
        self.notify.clone()
    }

    /// Start the worker pool. Spawns `concurrency` background tasks.
    /// Returns immediately — workers run until shutdown() is called.
    pub fn start(&self) -> Vec<tokio::task::JoinHandle<()>> {
        let semaphore = Arc::new(Semaphore::new(self.concurrency));
        let mut handles = Vec::new();

        for worker_id in 0..self.concurrency {
            let queue = self.job_queue.clone();
            let hg = self.hypergraph.clone();
            let engines = self.engines.clone();
            let sem = semaphore.clone();
            let notify = self.notify.clone();
            let mut shutdown_rx = self.shutdown_rx.clone();

            let handle = tokio::spawn(async move {
                loop {
                    // Check shutdown
                    if *shutdown_rx.borrow() {
                        tracing::debug!("Worker {} shutting down", worker_id);
                        break;
                    }

                    // Acquire semaphore permit
                    let _permit = match sem.acquire().await {
                        Ok(p) => p,
                        Err(_) => break, // Semaphore closed
                    };

                    // Try to dequeue a job
                    match queue.dequeue_next() {
                        Ok(Some(job)) => {
                            let job_id = job.id.clone();
                            tracing::info!("Worker {} processing job {}", worker_id, job_id);

                            // Mark running
                            if let Err(e) = queue.mark_running(&job_id) {
                                tracing::error!("Failed to mark job {} running: {}", job_id, e);
                                continue;
                            }

                            // Find engine — discriminant lookup so payload-bearing
                            // variants (e.g. SurrogateGeneration) match the registered
                            // sentinel without payload-equality.
                            if let Some(engine) = engines.get(&engine_key(&job.job_type)) {
                                let engine = engine.clone();
                                let hg = hg.clone();
                                let hg_enrich = hg.clone();
                                let queue = queue.clone();
                                let job_for_enrich = job.clone();

                                // Execute in blocking context (CPU-bound work)
                                let result =
                                    tokio::task::spawn_blocking(move || engine.execute(&job, &hg))
                                        .await;

                                match result {
                                    Ok(Ok(inference_result)) => {
                                        // Enrich entities/situations with results
                                        super::enrichment::enrich_from_result(
                                            &job_for_enrich,
                                            &inference_result,
                                            &hg_enrich,
                                        );
                                        if let Err(e) = queue.store_result(inference_result) {
                                            tracing::error!(
                                                "Failed to store result for {}: {}",
                                                job_id,
                                                e
                                            );
                                        }
                                        if let Err(e) = queue.mark_completed(&job_id) {
                                            tracing::error!(
                                                "Failed to mark {} completed: {}",
                                                job_id,
                                                e
                                            );
                                        }
                                    }
                                    Ok(Err(e)) => {
                                        tracing::error!("Job {} failed: {}", job_id, e);
                                        let _ = queue.mark_failed(&job_id, &e.to_string());
                                    }
                                    Err(e) => {
                                        tracing::error!("Job {} panicked: {}", job_id, e);
                                        let _ = queue.mark_failed(&job_id, "Worker task panicked");
                                    }
                                }
                            } else {
                                tracing::error!(
                                    "No engine registered for job type {:?}",
                                    job.job_type
                                );
                                let _ = queue.mark_failed(
                                    &job_id,
                                    &format!("No engine for {:?}", job.job_type),
                                );
                            }
                        }
                        Ok(None) => {
                            // No jobs available, wait for notification or timeout
                            tokio::select! {
                                _ = notify.notified() => {},
                                _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {},
                                _ = shutdown_rx.changed() => { break; }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Worker {} dequeue error: {}", worker_id, e);
                            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                        }
                    }
                }
            });

            handles.push(handle);
        }

        handles
    }

    /// Check whether an engine is registered for the given job type.
    ///
    /// Lookup is by discriminant, so a query for
    /// `SurrogateGeneration { source_narrative_id: Some("foo"), ... }` matches
    /// any registered SurrogateGeneration engine regardless of the engine's
    /// own sentinel field values.
    pub fn has_engine(&self, job_type: &InferenceJobType) -> bool {
        self.engines.contains_key(&engine_key(job_type))
    }

    /// Signal all workers to shut down.
    pub fn shutdown(&self) {
        let _ = self.shutdown.send(true);
        self.notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::inference::types::InferenceJob;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use uuid::Uuid;

    /// Mock engine that immediately returns a result.
    struct MockEngine {
        job_type: InferenceJobType,
    }

    impl InferenceEngine for MockEngine {
        fn job_type(&self) -> InferenceJobType {
            self.job_type.clone()
        }

        fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
            Ok(100)
        }

        fn execute(&self, job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<InferenceResult> {
            Ok(InferenceResult {
                job_id: job.id.clone(),
                job_type: job.job_type.clone(),
                target_id: job.target_id,
                result: serde_json::json!({"mock": true}),
                confidence: 0.9,
                explanation: None,
                status: JobStatus::Completed,
                created_at: job.created_at,
                completed_at: Some(Utc::now()),
            })
        }
    }

    /// Mock engine that fails.
    struct FailingEngine;

    impl InferenceEngine for FailingEngine {
        fn job_type(&self) -> InferenceJobType {
            InferenceJobType::AnomalyDetection
        }

        fn estimate_cost(&self, _job: &InferenceJob, _hypergraph: &Hypergraph) -> Result<u64> {
            Ok(100)
        }

        fn execute(
            &self,
            _job: &InferenceJob,
            _hypergraph: &Hypergraph,
        ) -> Result<InferenceResult> {
            Err(crate::TensaError::InferenceError(
                "Simulated failure".into(),
            ))
        }
    }

    fn make_job(job_type: InferenceJobType) -> InferenceJob {
        InferenceJob {
            id: Uuid::now_v7().to_string(),
            job_type,
            target_id: Uuid::now_v7(),
            parameters: serde_json::json!({}),
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            estimated_cost_ms: 100,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    #[tokio::test]
    async fn test_worker_processes_job() {
        let store = Arc::new(MemoryStore::new());
        let queue = Arc::new(JobQueue::new(store.clone()));
        let hg = Arc::new(Hypergraph::new(store));

        let mut pool = WorkerPool::new(queue.clone(), hg, 1);
        pool.register_engine(Arc::new(MockEngine {
            job_type: InferenceJobType::CausalDiscovery,
        }));

        let job = make_job(InferenceJobType::CausalDiscovery);
        let job_id = job.id.clone();
        queue.submit(job).unwrap();

        let handles = pool.start();
        // Give worker time to process
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        pool.shutdown();

        for h in handles {
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(2), h).await;
        }

        let completed = queue.get_job(&job_id).unwrap();
        assert_eq!(completed.status, JobStatus::Completed);

        let result = queue.get_result(&job_id).unwrap();
        assert_eq!(result.confidence, 0.9);
    }

    #[tokio::test]
    async fn test_worker_handles_failure() {
        let store = Arc::new(MemoryStore::new());
        let queue = Arc::new(JobQueue::new(store.clone()));
        let hg = Arc::new(Hypergraph::new(store));

        let mut pool = WorkerPool::new(queue.clone(), hg, 1);
        pool.register_engine(Arc::new(FailingEngine));

        let job = make_job(InferenceJobType::AnomalyDetection);
        let job_id = job.id.clone();
        queue.submit(job).unwrap();

        let handles = pool.start();
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        pool.shutdown();

        for h in handles {
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(2), h).await;
        }

        let failed = queue.get_job(&job_id).unwrap();
        assert_eq!(failed.status, JobStatus::Failed);
        assert!(failed.error.is_some());
    }

    #[tokio::test]
    async fn test_worker_notifier_wakes_workers() {
        let store = Arc::new(MemoryStore::new());
        let queue = Arc::new(JobQueue::new(store.clone()));
        let hg = Arc::new(Hypergraph::new(store));

        let mut pool = WorkerPool::new(queue.clone(), hg, 1);
        pool.register_engine(Arc::new(MockEngine {
            job_type: InferenceJobType::GameClassification,
        }));

        let notifier = pool.notifier();
        let handles = pool.start();

        // Submit after workers started
        let job = make_job(InferenceJobType::GameClassification);
        let job_id = job.id.clone();
        queue.submit(job).unwrap();
        notifier.notify_one(); // Wake worker

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        pool.shutdown();

        for h in handles {
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(2), h).await;
        }

        let completed = queue.get_job(&job_id).unwrap();
        assert_eq!(completed.status, JobStatus::Completed);
    }

    #[test]
    fn test_all_job_types_have_engines() {
        use crate::types::InferenceJobType;

        let store = Arc::new(MemoryStore::new());
        let hg = Arc::new(crate::Hypergraph::new(store.clone()));
        let jq = Arc::new(JobQueue::new(store));
        let mut pool = WorkerPool::new(jq, hg, 1);

        // Register all engines (mirrors main.rs registrations)
        pool.register_engine(Arc::new(crate::inference::causal::CausalEngine::default()));
        pool.register_engine(Arc::new(crate::inference::causal::CounterfactualEngine));
        pool.register_engine(Arc::new(crate::inference::causal::MissingLinksEngine));
        pool.register_engine(Arc::new(crate::inference::game::GameEngine::default()));
        pool.register_engine(Arc::new(
            crate::inference::motivation::MotivationEngine::default(),
        ));
        pool.register_engine(Arc::new(crate::analysis::centrality::CentralityEngine));
        pool.register_engine(Arc::new(crate::analysis::entropy::EntropyEngine));
        pool.register_engine(Arc::new(crate::analysis::beliefs::BeliefEngine));
        pool.register_engine(Arc::new(crate::analysis::evidence::EvidenceEngine));
        pool.register_engine(Arc::new(
            crate::analysis::argumentation::ArgumentationEngine,
        ));
        pool.register_engine(Arc::new(crate::analysis::contagion::ContagionEngine));
        pool.register_engine(Arc::new(crate::analysis::style_profile::StyleProfileEngine));
        pool.register_engine(Arc::new(
            crate::analysis::style_profile::StyleComparisonEngine,
        ));
        pool.register_engine(Arc::new(crate::analysis::style_profile::StyleAnomalyEngine));
        pool.register_engine(Arc::new(
            crate::analysis::pan_verification::AuthorshipVerificationEngine,
        ));
        pool.register_engine(Arc::new(
            crate::analysis::anomaly::AnomalyDetectionEngine::default(),
        ));
        pool.register_engine(Arc::new(crate::analysis::tcg::TCGAnomalyEngine));
        pool.register_engine(Arc::new(crate::inference::hawkes::HawkesEngine));
        pool.register_engine(Arc::new(crate::narrative::ArcEngine));
        pool.register_engine(Arc::new(crate::narrative::ActorArcEngine));
        pool.register_engine(Arc::new(crate::narrative::PatternMiningEngine));
        pool.register_engine(Arc::new(crate::narrative::MissingEventEngine));
        pool.register_engine(Arc::new(crate::inference::temporal_ilp::TemporalILPEngine));
        pool.register_engine(Arc::new(
            crate::inference::mean_field_game::MeanFieldGameEngine::default(),
        ));
        pool.register_engine(Arc::new(crate::analysis::psl::PslEngine));
        pool.register_engine(Arc::new(
            crate::inference::trajectory::TrajectoryEmbeddingEngine,
        ));
        pool.register_engine(Arc::new(
            crate::inference::simulation::NarrativeSimulationEngine::new(None),
        ));
        // Sprint 1: Core graph centrality engines
        pool.register_engine(Arc::new(crate::analysis::graph_centrality::PageRankEngine));
        pool.register_engine(Arc::new(
            crate::analysis::graph_centrality::EigenvectorEngine,
        ));
        pool.register_engine(Arc::new(crate::analysis::graph_centrality::HarmonicEngine));
        pool.register_engine(Arc::new(crate::analysis::graph_centrality::HitsEngine));
        // Sprint 2: Topology & community
        pool.register_engine(Arc::new(crate::analysis::topology::TopologyEngine));
        pool.register_engine(Arc::new(crate::analysis::topology::KCoreEngine));
        pool.register_engine(Arc::new(
            crate::analysis::community_detect::LabelPropagationEngine,
        ));
        // Sprint 7: Narrative-native
        pool.register_engine(Arc::new(
            crate::analysis::narrative_centrality::TemporalPageRankEngine,
        ));
        // Sprint 8: Temporal patterns
        pool.register_engine(Arc::new(
            crate::analysis::temporal_motifs::TemporalMotifEngine,
        ));
        pool.register_engine(Arc::new(crate::analysis::evolution::FactionEvolutionEngine));
        // Sprint 11: Graph embeddings & network inference
        pool.register_engine(Arc::new(crate::analysis::embeddings::FastRPEngine));
        pool.register_engine(Arc::new(crate::analysis::embeddings::Node2VecEngine));
        pool.register_engine(Arc::new(crate::inference::netinf::NetInfEngine));
        pool.register_engine(Arc::new(
            crate::analysis::narrative_centrality::CausalInfluenceEngine,
        ));
        pool.register_engine(Arc::new(
            crate::analysis::narrative_centrality::InfoBottleneckEngine,
        ));
        pool.register_engine(Arc::new(
            crate::analysis::narrative_centrality::AssortativityEngine,
        ));

        // EATH Phase 4: synth calibration + generation engines. Phase 7 adds
        // SurrogateSignificance to the all_types list + registers its engine.
        // Phase 7b adds SurrogateContagionSignificance.
        let synth_registry =
            std::sync::Arc::new(crate::synth::SurrogateRegistry::default());
        let (synth_cal, synth_gen, synth_hybrid) =
            crate::synth::engines::make_synth_engines(synth_registry.clone());
        pool.register_engine(synth_cal);
        pool.register_engine(synth_gen);
        pool.register_engine(synth_hybrid);
        pool.register_engine(std::sync::Arc::new(
            crate::synth::SurrogateSignificanceEngine::new(synth_registry.clone()),
        ));
        pool.register_engine(std::sync::Arc::new(
            crate::synth::SurrogateContagionSignificanceEngine::new(synth_registry.clone()),
        ));
        // EATH Extension Phase 13c — dual-null-model significance engine.
        pool.register_engine(std::sync::Arc::new(
            crate::synth::SurrogateDualSignificanceEngine::new(synth_registry.clone()),
        ));
        // EATH Extension Phase 14 — bistability-significance engine.
        pool.register_engine(std::sync::Arc::new(
            crate::synth::SurrogateBistabilitySignificanceEngine::new(synth_registry.clone()),
        ));
        // EATH Extension Phase 15b — SINDy hypergraph reconstruction engine.
        pool.register_engine(std::sync::Arc::new(
            crate::inference::hypergraph_reconstruction::ReconstructionEngine,
        ));
        // EATH Extension Phase 16c — opinion-dynamics-significance engine.
        pool.register_engine(std::sync::Arc::new(
            crate::synth::SurrogateOpinionSignificanceEngine::new(synth_registry),
        ));

        // Every InferenceJobType variant must have a registered engine
        let all_types = vec![
            InferenceJobType::CausalDiscovery,
            InferenceJobType::Counterfactual,
            InferenceJobType::MissingLinks,
            InferenceJobType::GameClassification,
            InferenceJobType::MotivationInference,
            InferenceJobType::AnomalyDetection,
            InferenceJobType::PatternMining,
            InferenceJobType::ArcClassification,
            InferenceJobType::ActorArcClassification,
            InferenceJobType::MissingEventPrediction,
            InferenceJobType::CentralityAnalysis,
            InferenceJobType::EntropyAnalysis,
            InferenceJobType::BeliefModeling,
            InferenceJobType::EvidenceCombination,
            InferenceJobType::ArgumentationAnalysis,
            InferenceJobType::ContagionAnalysis,
            InferenceJobType::StyleProfile,
            InferenceJobType::TCGAnomaly,
            InferenceJobType::NextEvent,
            InferenceJobType::StyleComparison,
            InferenceJobType::StyleAnomaly,
            InferenceJobType::AuthorshipVerification,
            InferenceJobType::TemporalILP,
            InferenceJobType::MeanFieldGame,
            InferenceJobType::ProbabilisticSoftLogic,
            InferenceJobType::TrajectoryEmbedding,
            InferenceJobType::NarrativeSimulation,
            InferenceJobType::PageRank,
            InferenceJobType::EigenvectorCentrality,
            InferenceJobType::HarmonicCentrality,
            InferenceJobType::HITS,
            InferenceJobType::Topology,
            InferenceJobType::LabelPropagation,
            InferenceJobType::KCore,
            InferenceJobType::TemporalPageRank,
            InferenceJobType::CausalInfluence,
            InferenceJobType::InfoBottleneck,
            InferenceJobType::Assortativity,
            InferenceJobType::TemporalMotifs,
            InferenceJobType::FactionEvolution,
            InferenceJobType::FastRP,
            InferenceJobType::Node2Vec,
            InferenceJobType::NetworkInference,
            // EATH Phase 4: synth calibration + generation. Sentinel field
            // values — the engine-coverage check uses discriminant lookup.
            // EATH Phase 7: SurrogateSignificance registered.
            // EATH Phase 7b: SurrogateContagionSignificance registered.
            InferenceJobType::SurrogateCalibration {
                narrative_id: String::new(),
                model: String::new(),
            },
            InferenceJobType::SurrogateGeneration {
                source_narrative_id: None,
                output_narrative_id: String::new(),
                model: String::new(),
                params_override: None,
                seed_override: None,
            },
            InferenceJobType::SurrogateSignificance {
                narrative_id: String::new(),
                metric_kind: String::new(),
                k: 0,
                model: String::new(),
            },
            InferenceJobType::SurrogateContagionSignificance {
                narrative_id: String::new(),
                k: 0,
                model: String::new(),
                contagion_params: serde_json::Value::Null,
            },
            // EATH Phase 9: hybrid (mixture-distribution) generation.
            InferenceJobType::SurrogateHybridGeneration {
                components: serde_json::Value::Null,
                output_narrative_id: String::new(),
                seed_override: None,
                num_steps: None,
            },
            // EATH Extension Phase 13c: dual-null-model significance.
            InferenceJobType::SurrogateDualSignificance {
                narrative_id: String::new(),
                metric: String::new(),
                k_per_model: 0,
                models: Vec::new(),
            },
            // EATH Extension Phase 14: bistability significance.
            InferenceJobType::SurrogateBistabilitySignificance {
                narrative_id: String::new(),
                params: serde_json::Value::Null,
                k: 0,
                models: Vec::new(),
            },
            // EATH Extension Phase 15b: SINDy hypergraph reconstruction.
            InferenceJobType::HypergraphReconstruction {
                narrative_id: String::new(),
                params: serde_json::Value::Null,
            },
            // EATH Extension Phase 16c: opinion-dynamics significance.
            InferenceJobType::SurrogateOpinionSignificance {
                narrative_id: String::new(),
                params: serde_json::Value::Null,
                k: 0,
                models: Vec::new(),
            },
        ];

        // Disinfo Sprint D1 (dual fingerprints) + Sprint D2 (spread velocity +
        // intervention) + Sprint D3 (CIB + superspreaders) — variants and
        // engines gated together so the engine-coverage invariant holds.
        #[cfg(feature = "disinfo")]
        let all_types = {
            let mut v = all_types;
            v.push(InferenceJobType::BehavioralFingerprint);
            v.push(InferenceJobType::DisinfoFingerprint);
            v.push(InferenceJobType::SpreadVelocity);
            v.push(InferenceJobType::SpreadIntervention);
            v.push(InferenceJobType::CibDetection);
            v.push(InferenceJobType::Superspreaders);
            pool.register_engine(Arc::new(
                crate::disinfo::engines::BehavioralFingerprintEngine,
            ));
            pool.register_engine(Arc::new(crate::disinfo::engines::DisinfoFingerprintEngine));
            pool.register_engine(Arc::new(crate::disinfo::engines::SpreadVelocityEngine));
            pool.register_engine(Arc::new(crate::disinfo::engines::SpreadInterventionEngine));
            pool.register_engine(Arc::new(crate::disinfo::engines::CibDetectionEngine));
            pool.register_engine(Arc::new(crate::disinfo::engines::SuperspreadersEngine));
            v
        };

        // Sprint D12: Adversarial narrative wargaming engines
        #[cfg(feature = "adversarial")]
        let all_types = {
            let mut v = all_types;
            v.push(InferenceJobType::AdversaryPolicy);
            v.push(InferenceJobType::CognitiveHierarchy);
            v.push(InferenceJobType::RewardFingerprint);
            v.push(InferenceJobType::CounterNarrative);
            v.push(InferenceJobType::Retrodiction);
            pool.register_engine(Arc::new(
                crate::adversarial::policy_gen::AdversaryPolicyEngine,
            ));
            pool.register_engine(Arc::new(
                crate::adversarial::policy_gen::CognitiveHierarchyEngine,
            ));
            pool.register_engine(Arc::new(
                crate::adversarial::reward_model::RewardFingerprintEngine,
            ));
            pool.register_engine(Arc::new(
                crate::adversarial::counter_gen::CounterNarrativeEngine,
            ));
            pool.register_engine(Arc::new(
                crate::adversarial::retrodiction::RetrodictionEngine,
            ));
            v
        };

        for jt in &all_types {
            assert!(
                pool.has_engine(jt),
                "No engine registered for {:?} — add it to main.rs and this test",
                jt
            );
        }

        // Also verify count matches (catches new variants added to enum but not registered)
        assert_eq!(
            all_types.len(),
            pool.engines.len(),
            "Engine count mismatch: test lists {} types but {} engines registered",
            all_types.len(),
            pool.engines.len()
        );
    }

    #[tokio::test]
    async fn test_shutdown_stops_workers() {
        let store = Arc::new(MemoryStore::new());
        let queue = Arc::new(JobQueue::new(store.clone()));
        let hg = Arc::new(Hypergraph::new(store));

        let pool = WorkerPool::new(queue, hg, 2);
        let handles = pool.start();

        pool.shutdown();

        for h in handles {
            let result = tokio::time::timeout(tokio::time::Duration::from_secs(3), h).await;
            assert!(result.is_ok(), "Worker should terminate after shutdown");
        }
    }
}
