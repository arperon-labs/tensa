#[cfg(feature = "server")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tensa=info".parse().unwrap()),
        )
        .init();

    tracing::info!("TENSA v{}", env!("CARGO_PKG_VERSION"));

    let store: std::sync::Arc<dyn tensa::KVStore> = {
        #[cfg(feature = "rocksdb")]
        {
            let data_dir =
                std::env::var("TENSA_DATA_DIR").unwrap_or_else(|_| "tensa_server_data".into());
            tracing::info!("RocksDB data dir: {}", data_dir);
            std::sync::Arc::new(tensa::store::rocks::RocksDBStore::open(&data_dir)?)
        }
        #[cfg(not(feature = "rocksdb"))]
        {
            tracing::warn!("RocksDB not available, using in-memory store");
            std::sync::Arc::new(tensa::store::memory::MemoryStore::new())
        }
    };

    let hypergraph = tensa::Hypergraph::new(store.clone());
    let interval_tree = tensa::temporal::index::IntervalTree::load(store.as_ref())
        .unwrap_or_else(|_| tensa::temporal::index::IntervalTree::new());

    let validation_queue = tensa::ingestion::queue::ValidationQueue::new(store.clone());
    let job_queue = std::sync::Arc::new(tensa::inference::jobs::JobQueue::new(store.clone()));

    // Reap stale `Running` jobs left over from a prior server instance.
    // At startup the worker pool is empty; any job still stamped `Running`
    // in KV is a zombie whose worker is gone (crashed / binary swapped / OS
    // reboot). Sweeping them to `Failed` unblocks dedup so fresh submissions
    // aren't deduped to a stuck predecessor.
    match job_queue.reap_stale_running() {
        Ok(0) => {}
        Ok(n) => tracing::info!("Reaped {} stale 'Running' job(s) from previous instance", n),
        Err(e) => tracing::warn!("Stale-job reaper failed: {}", e),
    }

    // Set up LLM config: env vars > persisted KV > None
    let env_llm_config = if let Ok(url) = std::env::var("LOCAL_LLM_URL") {
        let model = std::env::var("TENSA_MODEL").unwrap_or_else(|_| "default".to_string());
        let api_key = std::env::var("LOCAL_LLM_API_KEY").ok();
        tracing::info!(
            "LLM extraction enabled via local endpoint: {} (model: {})",
            url,
            model
        );
        Some(tensa::api::server::LlmConfig::Local {
            base_url: url,
            model,
            api_key,
        })
    } else if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        let model = match std::env::var("TENSA_MODEL") {
            Ok(m) => m,
            Err(_) => {
                tracing::warn!("OPENROUTER_API_KEY set but TENSA_MODEL not specified. Set TENSA_MODEL or configure model in Settings.");
                "x-ai/grok-3-mini-beta".to_string()
            }
        };
        tracing::info!("LLM extraction enabled via OpenRouter (model: {})", model);
        Some(tensa::api::server::LlmConfig::OpenRouter {
            api_key: key,
            model,
        })
    } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model =
            std::env::var("TENSA_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
        tracing::info!("LLM extraction enabled via Anthropic (model: {})", model);
        Some(tensa::api::server::LlmConfig::Anthropic {
            api_key: key,
            model,
        })
    } else {
        None
    };

    // Persisted config (from Settings UI) takes priority over env defaults
    let llm_config = if let Some(persisted) =
        tensa::api::settings_routes::load_persisted_llm_config(store.as_ref())
    {
        tracing::info!(
            "LLM config loaded from Settings (provider: {:?})",
            match &persisted {
                tensa::api::server::LlmConfig::OpenRouter { model, .. } =>
                    format!("openrouter/{}", model),
                tensa::api::server::LlmConfig::Local { model, .. } => format!("local/{}", model),
                tensa::api::server::LlmConfig::Anthropic { model, .. } =>
                    format!("anthropic/{}", model),
                tensa::api::server::LlmConfig::None => "none".to_string(),
                _ => "other".to_string(),
            }
        );
        persisted
    } else if let Some(cfg) = env_llm_config {
        cfg
    } else {
        tracing::warn!(
            "No LLM configured (set LOCAL_LLM_URL, OPENROUTER_API_KEY, or ANTHROPIC_API_KEY)"
        );
        tensa::api::server::LlmConfig::None
    };
    let extractor = llm_config.build_extractor();

    // Set up embedder: prefer ONNX model if available, else hash-based fallback
    type EmbedPair = (
        Option<std::sync::Arc<dyn tensa::ingestion::embed::EmbeddingProvider>>,
        Option<std::sync::Arc<std::sync::RwLock<tensa::ingestion::vector::VectorIndex>>>,
    );
    let hash_fallback = || -> EmbedPair {
        let dim = 64;
        // Load persisted vector index from KV, or create empty
        let vi = match tensa::ingestion::vector::VectorIndex::load(store.as_ref(), dim) {
            Ok(idx) => {
                tracing::info!("Loaded vector index from KV ({} vectors)", idx.len());
                idx
            }
            Err(_) => {
                tracing::info!("No persisted vector index found, starting empty");
                tensa::ingestion::vector::VectorIndex::new(dim)
            }
        };
        (
            Some(
                std::sync::Arc::new(tensa::ingestion::embed::HashEmbedding::new(dim))
                    as std::sync::Arc<dyn tensa::ingestion::embed::EmbeddingProvider>,
            ),
            Some(std::sync::Arc::new(std::sync::RwLock::new(vi))),
        )
    };

    let (embedder, vector_index): EmbedPair = {
        #[cfg(feature = "embedding")]
        {
            use tensa::ingestion::embed::EmbeddingProvider as _;
            match std::env::var("TENSA_EMBEDDING_MODEL") {
                Ok(model_dir) => {
                    match tensa::ingestion::embed::OnnxEmbedder::from_directory(&model_dir) {
                        Ok(embedder) => {
                            let dim = embedder.dimension();
                            tracing::info!(
                                "ONNX embedder loaded ({} dimensions) from {}",
                                dim,
                                model_dir
                            );
                            let vi = match tensa::ingestion::vector::VectorIndex::load(
                                store.as_ref(),
                                dim,
                            ) {
                                Ok(idx) => {
                                    tracing::info!(
                                        "Loaded vector index from KV ({} vectors)",
                                        idx.len()
                                    );
                                    idx
                                }
                                Err(_) => tensa::ingestion::vector::VectorIndex::new(dim),
                            };
                            (
                                Some(std::sync::Arc::new(embedder)
                                    as std::sync::Arc<
                                        dyn tensa::ingestion::embed::EmbeddingProvider,
                                    >),
                                Some(std::sync::Arc::new(std::sync::RwLock::new(vi))),
                            )
                        }
                        Err(err) => {
                            tracing::warn!(
                                "Failed to load ONNX model from {}: {}, falling back to hash embedding",
                                model_dir,
                                err
                            );
                            hash_fallback()
                        }
                    }
                }
                Err(_) => {
                    tracing::info!("No TENSA_EMBEDDING_MODEL set, using hash embedding (dim=64)");
                    hash_fallback()
                }
            }
        }
        #[cfg(not(feature = "embedding"))]
        {
            hash_fallback()
        }
    };

    // Load ingestion config: persisted KV > default from LLM config
    let ingestion_config = if let Some(persisted) =
        tensa::api::settings_routes::load_persisted_ingestion_config(store.as_ref())
    {
        tracing::info!(
            "Loaded persisted ingestion config from KV store (mode: {:?})",
            persisted.mode
        );
        persisted
    } else {
        tensa::ingestion::config::IngestionConfig {
            pass1: llm_config.clone(),
            ..Default::default()
        }
    };

    let ingestion_jobs = std::sync::Arc::new(tensa::ingestion::jobs::IngestionJobQueue::new(
        store.clone(),
    ));
    // Seed builtin ingestion templates on first run
    let _ = ingestion_jobs.init_builtin_templates();

    // Register default workspace metadata if not already present
    let default_ws_key = b"_ws/default";
    if store.get(default_ws_key).unwrap_or(None).is_none() {
        let meta = serde_json::json!({
            "id": "default",
            "name": "Default Workspace",
            "created_at": chrono::Utc::now(),
        });
        if let Ok(bytes) = serde_json::to_vec(&meta) {
            let _ = store.put(default_ws_key, &bytes);
        }
    }

    let state = std::sync::Arc::new(tensa::api::server::AppState {
        hypergraph,
        interval_tree: std::sync::RwLock::new(interval_tree),
        validation_queue,
        job_queue: Some(job_queue.clone()),
        extractor: std::sync::RwLock::new(extractor),
        llm_config: std::sync::RwLock::new(llm_config),
        embedder: std::sync::RwLock::new(embedder),
        embedder_model_name: std::sync::RwLock::new(
            std::env::var("TENSA_EMBEDDING_MODEL")
                .map(|p| p.split(['/', '\\']).last().unwrap_or("onnx").to_string())
                .unwrap_or_else(|_| "hash".to_string()),
        ),
        vector_index,
        ingestion_config: std::sync::RwLock::new(ingestion_config),
        inference_config: std::sync::RwLock::new(
            tensa::api::settings_routes::load_persisted_inference_config(store.as_ref())
                .unwrap_or_default(),
        ),
        job_watchers: std::sync::RwLock::new(std::collections::HashMap::new()),
        ingestion_jobs,
        ingestion_progress: std::sync::Mutex::new(std::collections::HashMap::new()),
        ingestion_cancel_flags: std::sync::Mutex::new(std::collections::HashMap::new()),
        llm_cache: Some(tensa::ingestion::llm_cache::LlmCache::new(store.clone())),
        doc_tracker: Some(tensa::ingestion::doc_status::DocStatusTracker::new(
            store.clone(),
        )),
        source_index: Some(tensa::ingestion::deletion::SourceIndex::new(store.clone())),
        rag_config: std::sync::RwLock::new(
            tensa::api::settings_routes::load_persisted_rag_config(store.as_ref())
                .unwrap_or_default(),
        ),
        reranker: None,
        root_store: store.clone(),
        geocoder: tensa::ingestion::geocode::Geocoder::new(store.clone()),
        inference_extractor: {
            let cfg =
                tensa::api::settings_routes::load_persisted_inference_llm_config(store.as_ref())
                    .unwrap_or(tensa::api::server::LlmConfig::None);
            std::sync::RwLock::new(cfg.build_extractor())
        },
        inference_llm_config: std::sync::RwLock::new(
            tensa::api::settings_routes::load_persisted_inference_llm_config(store.as_ref())
                .unwrap_or(tensa::api::server::LlmConfig::None),
        ),
        #[cfg(feature = "studio-chat")]
        chat_extractor: {
            let cfg = tensa::studio_chat::load_persisted_chat_llm_config(store.as_ref())
                .unwrap_or(tensa::api::server::LlmConfig::None);
            std::sync::RwLock::new(cfg.build_extractor())
        },
        #[cfg(feature = "studio-chat")]
        chat_llm_config: std::sync::RwLock::new(
            tensa::studio_chat::load_persisted_chat_llm_config(store.as_ref())
                .unwrap_or(tensa::api::server::LlmConfig::None),
        ),
        #[cfg(feature = "studio-chat")]
        chat_skills: tensa::studio_chat::SkillRegistry::default_bundled(),
        #[cfg(feature = "studio-chat")]
        chat_confirm_gate: std::sync::Arc::new(tensa::studio_chat::ConfirmGate::new()),
        #[cfg(feature = "studio-chat")]
        chat_mcp_proxies: {
            let set = std::sync::Arc::new(tensa::studio_chat::McpProxySet::new());
            let persisted = tensa::studio_chat::load_persisted_mcp_servers(store.as_ref());
            if !persisted.is_empty() {
                let set_clone = set.clone();
                tokio::spawn(async move { set_clone.sync(&persisted).await });
            }
            set
        },
        synth_registry: std::sync::Arc::new(tensa::synth::SurrogateRegistry::default()),
    });

    // Start inference worker pool with all engines registered.
    // Pool must live as long as the server — dropping it sends shutdown to workers.
    let worker_hg = std::sync::Arc::new(tensa::Hypergraph::new(store.clone()));
    let mut _worker_pool = tensa::inference::worker::WorkerPool::new(
        job_queue.clone(),
        worker_hg,
        2, // 2 concurrent workers
    );

    // Phase 2: Core inference engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::causal::CausalEngine::default(),
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::game::GameEngine::default(),
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::motivation::MotivationEngine::default(),
    ));

    // Phase 4: Analysis engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::centrality::CentralityEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::analysis::entropy::EntropyEngine));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::analysis::beliefs::BeliefEngine));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::evidence::EvidenceEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::argumentation::ArgumentationEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::contagion::ContagionEngine,
    ));

    // Narrative style engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::style_profile::StyleProfileEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::style_profile::StyleComparisonEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::style_profile::StyleAnomalyEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::pan_verification::AuthorshipVerificationEngine,
    ));

    // Narrative intelligence engines
    _worker_pool.register_engine(std::sync::Arc::new(tensa::narrative::ArcEngine));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::narrative::ActorArcEngine));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::narrative::PatternMiningEngine));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::narrative::MissingEventEngine));

    // Phase 2: Additional engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::causal::CounterfactualEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::causal::MissingLinksEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::anomaly::AnomalyDetectionEngine::default(),
    ));

    // Temporal Correlation Graph + Hawkes + Temporal ILP
    _worker_pool.register_engine(std::sync::Arc::new(tensa::analysis::tcg::TCGAnomalyEngine));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::inference::hawkes::HawkesEngine));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::temporal_ilp::TemporalILPEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::mean_field_game::MeanFieldGameEngine::default(),
    ));

    _worker_pool.register_engine(std::sync::Arc::new(tensa::analysis::psl::PslEngine));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::trajectory::TrajectoryEmbeddingEngine,
    ));

    // Graph embeddings + network inference (previously unregistered → "No engine for …")
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::embeddings::FastRPEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::embeddings::Node2VecEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::inference::netinf::NetInfEngine));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::simulation::NarrativeSimulationEngine::new(
            state.extractor.read().unwrap().clone(),
        ),
    ));

    #[cfg(feature = "generation")]
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::generation::job_engine::ChapterGenerationFitnessEngine::new(
            state.extractor.read().unwrap().clone(),
        ),
    ));

    // Sprint 1: Core graph centrality engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::graph_centrality::PageRankEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::graph_centrality::EigenvectorEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::graph_centrality::HarmonicEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::graph_centrality::HitsEngine,
    ));

    // Sprint 2: Topology & community engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::topology::TopologyEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(tensa::analysis::topology::KCoreEngine));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::community_detect::LabelPropagationEngine,
    ));

    // Sprint 7: Narrative-native engines
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::narrative_centrality::TemporalPageRankEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::narrative_centrality::CausalInfluenceEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::narrative_centrality::InfoBottleneckEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::narrative_centrality::AssortativityEngine,
    ));

    // Sprint 8: Temporal patterns
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::temporal_motifs::TemporalMotifEngine,
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::analysis::evolution::FactionEvolutionEngine,
    ));

    // EATH Phase 4: synth calibration + generation engines (one shared
    // registry between them — registry is read-only after construction).
    // Phase 7 adds SurrogateSignificanceEngine; Phase 7b adds
    // SurrogateContagionSignificanceEngine — all sharing the same registry.
    let synth_registry = std::sync::Arc::new(tensa::synth::SurrogateRegistry::default());
    let (synth_cal, synth_gen, synth_hybrid) =
        tensa::synth::engines::make_synth_engines(synth_registry.clone());
    _worker_pool.register_engine(synth_cal);
    _worker_pool.register_engine(synth_gen);
    _worker_pool.register_engine(synth_hybrid);
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::synth::SurrogateSignificanceEngine::new(synth_registry.clone()),
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::synth::SurrogateContagionSignificanceEngine::new(synth_registry.clone()),
    ));
    // EATH Extension Phase 13c + 14: dual-null + bistability significance.
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::synth::SurrogateDualSignificanceEngine::new(synth_registry.clone()),
    ));
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::synth::SurrogateBistabilitySignificanceEngine::new(synth_registry.clone()),
    ));

    // EATH Extension Phase 15b: SINDy hypergraph reconstruction engine.
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::inference::hypergraph_reconstruction::ReconstructionEngine,
    ));

    // EATH Extension Phase 16c: opinion-dynamics-significance engine.
    _worker_pool.register_engine(std::sync::Arc::new(
        tensa::synth::SurrogateOpinionSignificanceEngine::new(synth_registry),
    ));

    // Disinfo Sprint D1: dual fingerprint engines
    #[cfg(feature = "disinfo")]
    {
        _worker_pool.register_engine(std::sync::Arc::new(
            tensa::disinfo::engines::BehavioralFingerprintEngine,
        ));
        _worker_pool.register_engine(std::sync::Arc::new(
            tensa::disinfo::engines::DisinfoFingerprintEngine,
        ));
        _worker_pool.register_engine(std::sync::Arc::new(
            tensa::disinfo::engines::SpreadVelocityEngine,
        ));
        _worker_pool.register_engine(std::sync::Arc::new(
            tensa::disinfo::engines::SpreadInterventionEngine,
        ));
        // Sprint D3: CIB + superspreaders
        _worker_pool.register_engine(std::sync::Arc::new(
            tensa::disinfo::engines::CibDetectionEngine,
        ));
        _worker_pool.register_engine(std::sync::Arc::new(
            tensa::disinfo::engines::SuperspreadersEngine,
        ));
        // Sprint D4: Claims & fact-check pipeline
        _worker_pool.register_engine(std::sync::Arc::new(tensa::disinfo::ClaimOriginEngine));
        _worker_pool.register_engine(std::sync::Arc::new(tensa::disinfo::ClaimMatchEngine));
        // Sprint D5: Archetypes + DS fusion
        _worker_pool.register_engine(std::sync::Arc::new(tensa::disinfo::ArchetypeEngine));
        _worker_pool.register_engine(std::sync::Arc::new(tensa::disinfo::DisinfoAssessmentEngine));
    }

    let _worker_handles = _worker_pool.start();
    tracing::info!("Inference worker pool started");

    let addr = std::env::var("TENSA_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    tracing::info!("Starting API server on {}", addr);
    tensa::api::server::run(state, &addr).await
}

#[cfg(not(feature = "server"))]
fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tensa=info".parse().unwrap()),
        )
        .init();

    tracing::info!("TENSA v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Server feature not enabled. Build with: cargo run --features server");
}
