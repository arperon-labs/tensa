//! Disinformation extension modules.
//!
//! Sprint D1 (foundation): dual-fingerprint architecture — `BehavioralFingerprint`
//! (10 axes per actor) and `DisinformationFingerprint` (12 axes per narrative) —
//! plus per-task fingerprint comparison built on top of the existing per-layer
//! distance kernels in [`crate::analysis::similarity_metrics`] and bootstrap
//! confidence intervals in [`crate::analysis::stylometry_stats`].
//!
//! Subsequent disinfo sprints will:
//! - **D2**: extend SIR with SMIR + platform-aware β + VelocityMonitor; wire
//!   axes 1, 2, 11 of the disinfo fingerprint to real spread dynamics.
//! - **D3**: ship the CIB detection module; wire axis 7 (coordination_score).
//! - **D4**: claims & fact-check pipeline; wire axes 8, 9 (claim mutation,
//!   counter-narrative resistance).
//! - **D5**: archetype classification + DS disinfo signal fusion; wire axis 10
//!   (evidential_uncertainty).
//! - **D6**: multilingual + MCP client orchestrator + DISARM/MISP exports; wire
//!   axis 3 (linguistic_variance).
//!
//! Until those sprints land, the relevant axes are filled with `f64::NAN` and
//! the Studio renderer greys them out.

pub mod archetype_engine;
pub mod archetypes;
pub mod claims_engine;
pub mod comparison;
pub mod engines;
pub mod fingerprints;
pub mod fusion;
pub mod multilingual;
pub mod orchestrator;

pub use archetype_engine::{ArchetypeEngine, DisinfoAssessmentEngine};
pub use archetypes::{
    classify_actor_archetype, Archetype, ArchetypeDistribution, ArchetypeTemplate,
};
pub use claims_engine::{ClaimMatchEngine, ClaimOriginEngine};
pub use engines::{
    BehavioralFingerprintEngine, CibDetectionEngine, DisinfoFingerprintEngine,
    SpreadInterventionEngine, SpreadVelocityEngine, SuperspreadersEngine,
};

pub use comparison::{
    compare_fingerprints, AxisAnomaly, ComparisonKind, ComparisonTask, FingerprintComparison,
};
pub use fingerprints::{
    behavioral_axis_label, behavioral_axis_labels, behavioral_envelope,
    compute_behavioral_fingerprint, compute_disinfo_fingerprint, disinfo_axis_label,
    disinfo_axis_labels, disinfo_envelope, ensure_behavioral_fingerprint,
    ensure_disinfo_fingerprint, load_behavioral_fingerprint, load_disinfo_fingerprint,
    store_behavioral_fingerprint, store_disinfo_fingerprint, BehavioralAxis, BehavioralFingerprint,
    DisinfoAxis, DisinformationFingerprint, BEHAVIORAL_AXIS_COUNT, DISINFO_AXIS_COUNT,
};
pub use fusion::{
    fuse_disinfo_signals, DisinfoAssessment, DisinfoSignal, Hypothesis, SignalSource,
};
pub use multilingual::{
    detect_language, linguistic_variance, normalize_for_matching, strip_diacritics,
    transliterate_cyrillic_to_latin, transliterate_latin_to_cyrillic, LangDetectResult,
};
pub mod monitor;
pub use orchestrator::{
    list_source_health, load_source_health, post_to_hypergraph_items, record_poll_failure,
    record_poll_success, store_source_health, AuditEntry, McpSource, NormalizedPost, SourceHealth,
    SourceRegistry,
};
