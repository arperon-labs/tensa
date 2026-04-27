//! Dual-fingerprint structures for disinformation analysis.
//!
//! - [`BehavioralFingerprint`] — 10-axis per-actor metadata-derived vector
//!   used for CIB and bot detection.
//! - [`DisinformationFingerprint`] — 12-axis per-narrative composite combining
//!   spread dynamics, content factory output, source diversity, and DS
//!   evidential uncertainty.
//!
//! Many axes depend on disinfo-extension modules that ship in later sprints
//! (D2 spread, D3 CIB, D4 claims, D5 archetypes, D6 multilingual). Until then
//! those axes hold `f64::NAN`; the Studio renderer greys them out.
//!
//! Both fingerprints are persisted at the KV prefixes
//! [`crate::hypergraph::keys::BEHAVIORAL_FINGERPRINT`] (`bf/`) and
//! [`crate::hypergraph::keys::DISINFO_FINGERPRINT`] (`df/`), keyed by
//! `actor_uuid` and `narrative_id` respectively.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

use crate::analysis::analysis_key;
use crate::analysis::evidence::EvidenceResult;
use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::types::Situation;

/// Serialize a fixed-size array of `f64` mapping `NaN → JSON null`. JSON has
/// no NaN literal; this gives a stable wire format for axis vectors that may
/// be partially computed.
fn ser_nan_as_null<S, const N: usize>(arr: &[f64; N], s: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = s.serialize_seq(Some(N))?;
    for v in arr {
        if v.is_nan() {
            seq.serialize_element(&Option::<f64>::None)?;
        } else {
            seq.serialize_element(&Some(*v))?;
        }
    }
    seq.end()
}

/// Inverse of [`ser_nan_as_null`]: deserialize `null → NaN`.
fn de_null_as_nan<'de, D, const N: usize>(d: D) -> std::result::Result<[f64; N], D::Error>
where
    D: Deserializer<'de>,
{
    let opts: Vec<Option<f64>> = Vec::deserialize(d)?;
    if opts.len() != N {
        return Err(serde::de::Error::custom(format!(
            "expected fingerprint axis vector of length {}, got {}",
            N,
            opts.len()
        )));
    }
    let mut out = [f64::NAN; N];
    for (i, v) in opts.into_iter().enumerate() {
        out[i] = v.unwrap_or(f64::NAN);
    }
    Ok(out)
}

/// Number of axes in the behavioral fingerprint vector. Pinned constant —
/// changes here are versioning events.
pub const BEHAVIORAL_AXIS_COUNT: usize = 10;

/// Number of axes in the disinfo fingerprint vector. Pinned constant.
pub const DISINFO_AXIS_COUNT: usize = 12;

/// Stable index assignments for the behavioral fingerprint axes (per spec §3.2).
/// Ordering is part of the on-disk format — never reorder. `#[repr(u8)]` pins
/// the discriminant so `axis as usize` is well-defined.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehavioralAxis {
    PostingCadenceRegularity = 0,
    SleepPatternPresence = 1,
    EngagementRatio = 2,
    AccountMaturity = 3,
    PlatformDiversity = 4,
    ContentOriginality = 5,
    ResponseLatency = 6,
    HashtagConcentration = 7,
    NetworkInsularity = 8,
    TemporalCoordination = 9,
}

/// Stable index assignments for the disinfo fingerprint axes (per spec §3.1).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisinfoAxis {
    ViralityVelocity = 0,
    CrossPlatformJumpRate = 1,
    LinguisticVariance = 2,
    BotAmplificationRatio = 3,
    EmotionalLoading = 4,
    SourceDiversity = 5,
    CoordinationScore = 6,
    ClaimMutationRate = 7,
    CounterNarrativeResistance = 8,
    EvidentialUncertainty = 9,
    TemporalAnomaly = 10,
    AuthorityExploitation = 11,
}

/// Single source of truth for behavioral axis labels — indexed by axis discriminant.
const BEHAVIORAL_LABELS: [&str; BEHAVIORAL_AXIS_COUNT] = [
    "posting_cadence_regularity",
    "sleep_pattern_presence",
    "engagement_ratio",
    "account_maturity",
    "platform_diversity",
    "content_originality",
    "response_latency",
    "hashtag_concentration",
    "network_insularity",
    "temporal_coordination",
];

/// Single source of truth for disinfo axis labels — indexed by axis discriminant.
const DISINFO_LABELS: [&str; DISINFO_AXIS_COUNT] = [
    "virality_velocity",
    "cross_platform_jump_rate",
    "linguistic_variance",
    "bot_amplification_ratio",
    "emotional_loading",
    "source_diversity",
    "coordination_score",
    "claim_mutation_rate",
    "counter_narrative_resistance",
    "evidential_uncertainty",
    "temporal_anomaly",
    "authority_exploitation",
];

/// Human-readable label for a behavioral axis (for Studio + JSON responses).
pub fn behavioral_axis_label(axis: BehavioralAxis) -> &'static str {
    BEHAVIORAL_LABELS[axis as usize]
}

/// Ordered list of all behavioral axis labels (length matches `BEHAVIORAL_AXIS_COUNT`).
pub fn behavioral_axis_labels() -> [&'static str; BEHAVIORAL_AXIS_COUNT] {
    BEHAVIORAL_LABELS
}

/// Human-readable label for a disinfo axis.
pub fn disinfo_axis_label(axis: DisinfoAxis) -> &'static str {
    DISINFO_LABELS[axis as usize]
}

/// Ordered list of all disinfo axis labels (length matches `DISINFO_AXIS_COUNT`).
pub fn disinfo_axis_labels() -> [&'static str; DISINFO_AXIS_COUNT] {
    DISINFO_LABELS
}

/// Behavioral fingerprint for a single actor (account/entity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralFingerprint {
    pub actor_id: Uuid,
    /// 10 axes, indexed by `BehavioralAxis`. `f64::NAN` = not yet computed
    /// (serialized as JSON `null` and deserialized back to NaN).
    #[serde(
        serialize_with = "ser_nan_as_null::<_, BEHAVIORAL_AXIS_COUNT>",
        deserialize_with = "de_null_as_nan::<_, BEHAVIORAL_AXIS_COUNT>"
    )]
    pub axes: [f64; BEHAVIORAL_AXIS_COUNT],
    pub computed_at: DateTime<Utc>,
    /// Number of situations the entity participated in that informed this fingerprint.
    pub sample_size: usize,
}

impl BehavioralFingerprint {
    /// Construct an all-NaN fingerprint for an actor. Useful as a base
    /// before partial computation.
    pub fn empty(actor_id: Uuid) -> Self {
        Self {
            actor_id,
            axes: [f64::NAN; BEHAVIORAL_AXIS_COUNT],
            computed_at: Utc::now(),
            sample_size: 0,
        }
    }

    /// Read a single axis value.
    pub fn axis(&self, axis: BehavioralAxis) -> f64 {
        self.axes[axis as usize]
    }

    /// Set a single axis value.
    pub fn set_axis(&mut self, axis: BehavioralAxis, value: f64) {
        self.axes[axis as usize] = value;
    }

    /// Number of axes with non-NaN values — used by the Studio renderer.
    pub fn computed_axes(&self) -> usize {
        self.axes.iter().filter(|v| !v.is_nan()).count()
    }

    /// Fraction of non-NaN axes (0.0–1.0).
    pub fn completeness(&self) -> f64 {
        self.computed_axes() as f64 / BEHAVIORAL_AXIS_COUNT as f64
    }
}

/// Disinformation fingerprint for a single narrative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisinformationFingerprint {
    pub narrative_id: String,
    /// 12 axes, indexed by `DisinfoAxis`. `f64::NAN` = not yet computed
    /// (serialized as JSON `null` and deserialized back to NaN).
    #[serde(
        serialize_with = "ser_nan_as_null::<_, DISINFO_AXIS_COUNT>",
        deserialize_with = "de_null_as_nan::<_, DISINFO_AXIS_COUNT>"
    )]
    pub axes: [f64; DISINFO_AXIS_COUNT],
    pub computed_at: DateTime<Utc>,
    /// Number of situations in the narrative that informed this fingerprint.
    pub sample_size: usize,
}

impl DisinformationFingerprint {
    pub fn empty(narrative_id: impl Into<String>) -> Self {
        Self {
            narrative_id: narrative_id.into(),
            axes: [f64::NAN; DISINFO_AXIS_COUNT],
            computed_at: Utc::now(),
            sample_size: 0,
        }
    }

    pub fn axis(&self, axis: DisinfoAxis) -> f64 {
        self.axes[axis as usize]
    }

    pub fn set_axis(&mut self, axis: DisinfoAxis, value: f64) {
        self.axes[axis as usize] = value;
    }

    pub fn computed_axes(&self) -> usize {
        self.axes.iter().filter(|v| !v.is_nan()).count()
    }

    /// Fraction of non-NaN axes (0.0–1.0).
    pub fn completeness(&self) -> f64 {
        self.computed_axes() as f64 / DISINFO_AXIS_COUNT as f64
    }
}

// ─── Behavioral Fingerprint Computation ─────────────────────

/// Compute the behavioral fingerprint for an actor from observable hypergraph
/// state. Axes that depend on later-sprint modules (account_maturity,
/// network_insularity, temporal_coordination) are left as NaN.
pub fn compute_behavioral_fingerprint(
    hypergraph: &Hypergraph,
    actor_id: &Uuid,
) -> Result<BehavioralFingerprint> {
    let mut fp = BehavioralFingerprint::empty(*actor_id);
    let entity = hypergraph.get_entity(actor_id)?;

    // Load every situation this entity participates in exactly once.
    // Downstream helpers receive the slice rather than re-reading from KV.
    let participations = hypergraph.get_situations_for_entity(actor_id)?;
    let mut seen: HashSet<Uuid> = HashSet::with_capacity(participations.len());
    let mut situations: Vec<Situation> = Vec::with_capacity(participations.len());
    for p in &participations {
        if !seen.insert(p.situation_id) {
            continue;
        }
        if let Ok(sit) = hypergraph.get_situation(&p.situation_id) {
            situations.push(sit);
        }
    }
    let mut sit_times: Vec<DateTime<Utc>> =
        situations.iter().filter_map(|s| s.temporal.start).collect();
    sit_times.sort();
    fp.sample_size = situations.len();

    if sit_times.len() >= 3 {
        let intervals_secs: Vec<f64> = sit_times
            .windows(2)
            .map(|w| (w[1] - w[0]).num_seconds().max(0) as f64)
            .collect();
        fp.set_axis(
            BehavioralAxis::PostingCadenceRegularity,
            cadence_regularity(&intervals_secs),
        );
    }

    if !sit_times.is_empty() {
        fp.set_axis(
            BehavioralAxis::SleepPatternPresence,
            sleep_pattern_presence(&sit_times),
        );
    }

    if let Some(ratio) = engagement_ratio(&entity.properties) {
        fp.set_axis(BehavioralAxis::EngagementRatio, ratio);
    }

    // Account maturity: prefer properties.account_created_at, else entity.created_at.
    if let Some(maturity) = account_maturity(&entity) {
        fp.set_axis(BehavioralAxis::AccountMaturity, maturity);
    }

    fp.set_axis(
        BehavioralAxis::PlatformDiversity,
        platform_diversity(&entity.properties),
    );

    if !situations.is_empty() {
        fp.set_axis(
            BehavioralAxis::ContentOriginality,
            content_originality(&situations),
        );
    }

    // Mean inter-arrival mapped to [0, 1] via exp(-mean_hours / 24).
    if sit_times.len() >= 2 {
        let mean_secs: f64 = sit_times
            .windows(2)
            .map(|w| (w[1] - w[0]).num_seconds().max(0) as f64)
            .sum::<f64>()
            / (sit_times.len() - 1) as f64;
        let mean_hours = mean_secs / 3600.0;
        fp.set_axis(
            BehavioralAxis::ResponseLatency,
            (-mean_hours / 24.0).exp().clamp(0.0, 1.0),
        );
    }

    let hashtag_h = hashtag_concentration(&situations);
    if !hashtag_h.is_nan() {
        fp.set_axis(BehavioralAxis::HashtagConcentration, hashtag_h);
    }

    // network_insularity (#8) and temporal_coordination (#9) need CIB —
    // wired in Sprint D3.

    Ok(fp)
}

fn cadence_regularity(intervals_secs: &[f64]) -> f64 {
    if intervals_secs.len() < 2 {
        return f64::NAN;
    }
    let n = intervals_secs.len() as f64;
    let mean: f64 = intervals_secs.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var: f64 = intervals_secs
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / n;
    let cv = var.sqrt() / mean;
    // Map CV to [0, 1] regularity score: CV=0 → 1.0 perfectly regular,
    // CV→∞ → 0.0 chaotic. Sigmoid-like via 1 / (1 + cv).
    (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
}

fn sleep_pattern_presence(times: &[DateTime<Utc>]) -> f64 {
    if times.is_empty() {
        return f64::NAN;
    }
    let sleep_count = times
        .iter()
        .filter(|t| {
            let h = t.hour();
            h < 6 // 00:00–05:59 UTC nominal sleep window
        })
        .count();
    let sleep_ratio = sleep_count as f64 / times.len() as f64;
    // Expected human ratio for a UTC-anywhere account is ~6/24 = 0.25 if active
    // continuously. Bots show much higher ratios. Map "no sleep gap" to 1.0:
    // observed_ratio / expected_human_ratio, clamped.
    (sleep_ratio / 0.25).clamp(0.0, 1.0)
}

fn engagement_ratio(properties: &serde_json::Value) -> Option<f64> {
    let eng = properties.get("engagement")?;
    let likes = eng.get("likes").and_then(|v| v.as_u64()).unwrap_or(0);
    let shares = eng.get("shares").and_then(|v| v.as_u64()).unwrap_or(0);
    if likes == 0 {
        return Some(0.0);
    }
    Some(((shares as f64) / (likes as f64)).min(1.0).max(0.0))
}

fn account_maturity(entity: &crate::types::Entity) -> Option<f64> {
    // Prefer explicit account_created_at, fall back to entity.created_at.
    let created = entity
        .properties
        .get("account_created_at")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or(entity.created_at);
    let age = Utc::now().signed_duration_since(created);
    if age < Duration::zero() {
        return Some(0.0);
    }
    let days = age.num_days() as f64;
    Some(1.0 - (-days / 365.0).exp())
}

fn platform_diversity(properties: &serde_json::Value) -> f64 {
    let mut platforms: HashSet<String> = HashSet::new();
    if let Some(arr) = properties.get("platforms").and_then(|v| v.as_array()) {
        for p in arr {
            if let Some(s) = p.as_str() {
                platforms.insert(s.to_lowercase());
            }
        }
    }
    if let Some(s) = properties.get("platform").and_then(|v| v.as_str()) {
        platforms.insert(s.to_lowercase());
    }
    if platforms.is_empty() {
        return 0.0;
    }
    (platforms.len() as f64 / 4.0).tanh().clamp(0.0, 1.0)
}

fn content_originality(situations: &[Situation]) -> f64 {
    if situations.is_empty() {
        return f64::NAN;
    }
    let original = situations
        .iter()
        .filter(|sit| {
            !sit.raw_content
                .iter()
                .any(|cb| cb.content.starts_with("RT @") || cb.content.starts_with("repost:"))
        })
        .count();
    original as f64 / situations.len() as f64
}

fn hashtag_re() -> &'static regex::Regex {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"#(\w+)").expect("hashtag regex"))
}

fn hashtag_concentration(situations: &[Situation]) -> f64 {
    let mut counts: HashMap<String, u64> = HashMap::new();
    let re = hashtag_re();
    for sit in situations {
        for cb in &sit.raw_content {
            for cap in re.captures_iter(&cb.content) {
                if let Some(m) = cap.get(1) {
                    *counts.entry(m.as_str().to_lowercase()).or_insert(0) += 1;
                }
            }
        }
    }
    let total: u64 = counts.values().sum();
    if total == 0 {
        return f64::NAN;
    }
    let h: f64 = counts
        .values()
        .map(|&c| {
            let p = c as f64 / total as f64;
            p * p
        })
        .sum();
    h.clamp(0.0, 1.0)
}

// ─── Disinfo Fingerprint Computation ────────────────────────

/// Compute a partial disinfo fingerprint for a narrative. Sprint D1 wires only
/// `source_diversity` (axis 5) and `evidential_uncertainty` (axis 9, when
/// evidence records exist). Other axes are populated by future sprints.
pub fn compute_disinfo_fingerprint(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<DisinformationFingerprint> {
    let mut fp = DisinformationFingerprint::empty(narrative_id);
    let situations = hypergraph.list_situations_by_narrative(narrative_id)?;
    fp.sample_size = situations.len();

    // Axis 5: source diversity — inverse Herfindahl over distinct sources
    // attributing entities/situations in the narrative.
    if let Some(div) = source_diversity(hypergraph, narrative_id, &situations)? {
        fp.set_axis(DisinfoAxis::SourceDiversity, div);
    }

    // Axis 9: evidential uncertainty — average plausibility-belief gap across
    // any persisted DS evidence records for this narrative.
    if let Some(unc) = evidential_uncertainty(hypergraph, narrative_id)? {
        fp.set_axis(DisinfoAxis::EvidentialUncertainty, unc);
    }

    // Sprint D2 — spread dynamics. Wires three axes from the persisted SMIR
    // result + cross-platform jump log + most-recent velocity alert. Each
    // helper returns None when the corresponding D2 module hasn't been run
    // yet; the axis simply stays NaN, consistent with Sprint D1 semantics.
    if let Some(virality) = virality_velocity(hypergraph, narrative_id)? {
        fp.set_axis(DisinfoAxis::ViralityVelocity, virality);
    }
    if let Some(jump) = cross_platform_jump_rate(hypergraph, narrative_id)? {
        fp.set_axis(DisinfoAxis::CrossPlatformJumpRate, jump);
    }
    if let Some(anom) = temporal_anomaly(hypergraph, narrative_id)? {
        fp.set_axis(DisinfoAxis::TemporalAnomaly, anom);
    }

    // Sprint D3 — CIB detection. Wires axis 7 (coordination_score) from the
    // persisted CIB clusters for this narrative. Returns None when the D3
    // `INFER CIB` job has never run; the axis stays NaN in that case.
    if let Some(coord) = coordination_score(hypergraph, narrative_id)? {
        fp.set_axis(DisinfoAxis::CoordinationScore, coord);
    }

    // Sprint D4 — claims & fact-check pipeline. Wires axes 8 and 9 from
    // the claims module. Returns None when no claims have been detected.
    #[cfg(feature = "disinfo")]
    {
        if let Some(rate) =
            crate::claims::mutation::narrative_mutation_rate(hypergraph, narrative_id)?
        {
            fp.set_axis(DisinfoAxis::ClaimMutationRate, rate);
        }
        if let Some(resist) =
            crate::claims::fact_check::counter_narrative_resistance(hypergraph, narrative_id)?
        {
            fp.set_axis(DisinfoAxis::CounterNarrativeResistance, resist);
        }
    }

    // Sprint D6 — multilingual. Wires axis 3 (linguistic_variance) from
    // language distribution across situations in the narrative.
    #[cfg(feature = "disinfo")]
    {
        if let Some(lv) = linguistic_variance_axis(hypergraph, narrative_id, &situations)? {
            fp.set_axis(DisinfoAxis::LinguisticVariance, lv);
        }
    }

    // Axes 4 — bot_amplification_ratio — wired in future sprints.

    Ok(fp)
}

/// Axis #7 — `coordination_score`. Computed from the persisted CIB clusters
/// as the fraction of actors involved in any flagged cluster, weighted by
/// mean within-cluster density. Returns None when no CIB run has occurred.
fn coordination_score(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Option<f64>> {
    let clusters = crate::analysis::cib::list_clusters(hypergraph, narrative_id)?;
    if clusters.is_empty() {
        return Ok(None);
    }
    let total_actors = hypergraph
        .list_entities_by_narrative(narrative_id)?
        .into_iter()
        .filter(|e| matches!(e.entity_type, crate::types::EntityType::Actor))
        .count();
    if total_actors == 0 {
        return Ok(None);
    }
    // Union of members across all flagged clusters.
    let mut involved: HashSet<Uuid> = HashSet::new();
    let mut density_weight = 0.0;
    for c in &clusters {
        for m in &c.members {
            involved.insert(*m);
        }
        density_weight += c.density;
    }
    let coverage = involved.len() as f64 / total_actors as f64;
    let mean_density = density_weight / clusters.len() as f64;
    Ok(Some((coverage * mean_density).clamp(0.0, 1.0)))
}

// ─── Sprint D2 axis wiring ────────────────────────────────────

/// Axis #1 — `virality_velocity`. Sigmoid of (R₀ − 1.0) using the persisted
/// SMIR result. Returns None when D2's `run_smir_contagion` has never run.
fn virality_velocity(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Option<f64>> {
    let result = match crate::analysis::contagion::load_smir_result(hypergraph, narrative_id)? {
        Some(r) => r,
        None => return Ok(None),
    };
    let x = result.r0_overall - 1.0;
    let sig = 1.0 / (1.0 + (-x).exp());
    Ok(Some(sig.clamp(0.0, 1.0)))
}

/// Axis #2 — `cross_platform_jump_rate`. Number of distinct (from, to)
/// platform pairs observed in the narrative's jump log, normalized by the
/// theoretical maximum. Returns None when no jumps have been detected.
fn cross_platform_jump_rate(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Option<f64>> {
    let jumps = crate::analysis::contagion::list_cross_platform_jumps(hypergraph, narrative_id)?;
    if jumps.is_empty() {
        return Ok(None);
    }
    let mut platforms: HashSet<String> = HashSet::new();
    for jump in &jumps {
        platforms.insert(jump.from_platform.clone());
        platforms.insert(jump.to_platform.clone());
    }
    let n = platforms.len();
    // Max distinct ordered pairs over n platforms is n*(n-1).
    let max_pairs = (n.saturating_mul(n.saturating_sub(1))).max(1) as f64;
    let observed = jumps.len() as f64;
    Ok(Some((observed / max_pairs).clamp(0.0, 1.0)))
}

/// Axis #11 — `temporal_anomaly`. Most-recent velocity alert's z-score
/// normalized via tanh(z/4) so a 2σ alert sits around 0.46, a 4σ alert near
/// 0.76, never saturating. None when no alerts exist.
fn temporal_anomaly(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Option<f64>> {
    let monitor = crate::analysis::velocity_monitor::VelocityMonitor::new(hypergraph);
    let alerts = monitor.recent_alerts(narrative_id, 1)?;
    let alert = match alerts.into_iter().next() {
        Some(a) => a,
        None => return Ok(None),
    };
    let z = alert.z_score.abs();
    Ok(Some((z / 4.0).tanh().clamp(0.0, 1.0)))
}

fn source_diversity(
    hypergraph: &Hypergraph,
    _narrative_id: &str,
    situations: &[Situation],
) -> Result<Option<f64>> {
    let mut source_counts: HashMap<String, u64> = HashMap::new();
    let store = hypergraph.store();
    for sit in situations {
        let prefix = keys::source_attribution_reverse_prefix(&sit.id);
        let pairs = store
            .prefix_scan(&prefix)
            .map_err(|e| TensaError::Internal(e.to_string()))?;
        for (key, _) in pairs {
            if key.len() >= 16 {
                // Canonical UUID string so the SourceAttribution path and the
                // SourceReference.source_id provenance path collide on the same
                // source instead of double-counting.
                if let Ok(uuid) = Uuid::from_slice(&key[key.len() - 16..]) {
                    *source_counts.entry(uuid.to_string()).or_insert(0) += 1;
                }
            }
        }
        for prov in &sit.provenance {
            if let Some(sid) = &prov.source_id {
                *source_counts.entry(sid.clone()).or_insert(0) += 1;
            }
        }
    }
    let total: u64 = source_counts.values().sum();
    if total == 0 {
        return Ok(None);
    }
    let h: f64 = source_counts
        .values()
        .map(|&c| {
            let p = c as f64 / total as f64;
            p * p
        })
        .sum();
    let n = source_counts.len() as f64;
    if n <= 1.0 {
        return Ok(Some(0.0));
    }
    // Effective-number-of-sources (1/H) normalized by max possible distinct.
    Ok(Some(((1.0 / h) / n).clamp(0.0, 1.0)))
}

fn evidential_uncertainty(hypergraph: &Hypergraph, narrative_id: &str) -> Result<Option<f64>> {
    let prefix_str = std::str::from_utf8(keys::ANALYSIS_EVIDENCE).unwrap_or("");
    let prefix = format!("{prefix_str}{narrative_id}/");
    let pairs = hypergraph
        .store()
        .prefix_scan(prefix.as_bytes())
        .map_err(|e| TensaError::Internal(e.to_string()))?;
    if pairs.is_empty() {
        return Ok(None);
    }
    let mut gaps: Vec<f64> = Vec::with_capacity(pairs.len());
    for (_, value) in pairs {
        // EvidenceResult exposes per-hypothesis intervals as `belief_plausibility`
        // (see crate::analysis::evidence::EvidenceResult). Each `BeliefPlausibility`
        // already carries `uncertainty = plausibility − belief`.
        let parsed: EvidenceResult = match serde_json::from_slice(&value) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for bp in &parsed.belief_plausibility {
            gaps.push(bp.uncertainty.max(0.0));
        }
    }
    if gaps.is_empty() {
        return Ok(None);
    }
    let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
    Ok(Some(mean_gap.clamp(0.0, 1.0)))
}

// ─── Sprint D6 axis wiring ─────────────��─────────────────────

/// Axis #3 — `linguistic_variance`. Shannon entropy of the language distribution
/// across situations in the narrative, normalized to [0, 1].
///
/// Extracts language information from each situation's raw content via
/// [`super::multilingual::detect_language`] and delegates to
/// [`super::multilingual::linguistic_variance`].
///
/// Returns `None` when no situations have classifiable text (all unknown).
#[cfg(feature = "disinfo")]
fn linguistic_variance_axis(
    _hypergraph: &Hypergraph,
    _narrative_id: &str,
    situations: &[crate::types::Situation],
) -> Result<Option<f64>> {
    if situations.is_empty() {
        return Ok(None);
    }
    let languages: Vec<String> = situations
        .iter()
        .filter_map(|sit| {
            // Prefer an explicit language tag in situation properties/metadata.
            // Fall back to running language detection on the first content block.
            let content_text = sit
                .raw_content
                .first()
                .map(|b| b.content.as_str())
                .unwrap_or("");
            if content_text.trim().is_empty() {
                return None;
            }
            let detected = super::multilingual::detect_language(content_text);
            if detected.language == "unknown" || detected.confidence < 0.3 {
                return None;
            }
            Some(detected.language)
        })
        .collect();
    if languages.is_empty() {
        return Ok(None);
    }
    let variance = super::multilingual::linguistic_variance(&languages);
    Ok(Some(variance))
}

// ─── Persistence ────────────────────────────────────────────

fn behavioral_key(actor_id: &Uuid) -> Vec<u8> {
    analysis_key(keys::BEHAVIORAL_FINGERPRINT, &[&actor_id.to_string()])
}

fn disinfo_key(narrative_id: &str) -> Vec<u8> {
    analysis_key(keys::DISINFO_FINGERPRINT, &[narrative_id])
}

fn kv_put<T: Serialize>(hg: &Hypergraph, key: &[u8], value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|e| TensaError::Serialization(e.to_string()))?;
    hg.store().put(key, &bytes)
}

fn kv_load<T: serde::de::DeserializeOwned>(hg: &Hypergraph, key: &[u8]) -> Result<Option<T>> {
    match hg.store().get(key)? {
        Some(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).map_err(|e| TensaError::Serialization(e.to_string()))?,
        )),
        None => Ok(None),
    }
}

pub fn store_behavioral_fingerprint(
    hypergraph: &Hypergraph,
    fp: &BehavioralFingerprint,
) -> Result<()> {
    kv_put(hypergraph, &behavioral_key(&fp.actor_id), fp)
}

pub fn load_behavioral_fingerprint(
    hypergraph: &Hypergraph,
    actor_id: &Uuid,
) -> Result<Option<BehavioralFingerprint>> {
    kv_load(hypergraph, &behavioral_key(actor_id))
}

pub fn store_disinfo_fingerprint(
    hypergraph: &Hypergraph,
    fp: &DisinformationFingerprint,
) -> Result<()> {
    kv_put(hypergraph, &disinfo_key(&fp.narrative_id), fp)
}

pub fn load_disinfo_fingerprint(
    hypergraph: &Hypergraph,
    narrative_id: &str,
) -> Result<Option<DisinformationFingerprint>> {
    kv_load(hypergraph, &disinfo_key(narrative_id))
}

/// Load the cached behavioral fingerprint, or compute + persist it if absent.
/// Pass `force = true` to recompute regardless of cache state.
pub fn ensure_behavioral_fingerprint(
    hypergraph: &Hypergraph,
    actor_id: &Uuid,
    force: bool,
) -> Result<BehavioralFingerprint> {
    if !force {
        if let Some(fp) = load_behavioral_fingerprint(hypergraph, actor_id)? {
            return Ok(fp);
        }
    }
    let fresh = compute_behavioral_fingerprint(hypergraph, actor_id)?;
    store_behavioral_fingerprint(hypergraph, &fresh)?;
    Ok(fresh)
}

/// Load the cached disinfo fingerprint, or compute + persist it if absent.
pub fn ensure_disinfo_fingerprint(
    hypergraph: &Hypergraph,
    narrative_id: &str,
    force: bool,
) -> Result<DisinformationFingerprint> {
    if !force {
        if let Some(fp) = load_disinfo_fingerprint(hypergraph, narrative_id)? {
            return Ok(fp);
        }
    }
    let fresh = compute_disinfo_fingerprint(hypergraph, narrative_id)?;
    store_disinfo_fingerprint(hypergraph, &fresh)?;
    Ok(fresh)
}

/// JSON envelope for a behavioral fingerprint response (Studio + MCP).
pub fn behavioral_envelope(fp: &BehavioralFingerprint) -> serde_json::Value {
    serde_json::json!({ "fingerprint": fp, "axis_labels": behavioral_axis_labels() })
}

/// JSON envelope for a disinfo fingerprint response (Studio + MCP).
pub fn disinfo_envelope(fp: &DisinformationFingerprint) -> serde_json::Value {
    serde_json::json!({ "fingerprint": fp, "axis_labels": disinfo_axis_labels() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::test_helpers::{add_entity, link, make_hg};
    use crate::types::*;
    use chrono::TimeZone;

    #[test]
    fn axis_label_arrays_match_count() {
        assert_eq!(behavioral_axis_labels().len(), BEHAVIORAL_AXIS_COUNT);
        assert_eq!(disinfo_axis_labels().len(), DISINFO_AXIS_COUNT);
    }

    #[test]
    fn empty_fingerprint_is_all_nan() {
        let fp = BehavioralFingerprint::empty(Uuid::nil());
        assert_eq!(fp.computed_axes(), 0);
        assert!(fp.axes.iter().all(|v| v.is_nan()));

        let dfp = DisinformationFingerprint::empty("test-narr");
        assert_eq!(dfp.computed_axes(), 0);
        assert!(dfp.axes.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn cadence_regularity_perfect() {
        let intervals = vec![60.0, 60.0, 60.0, 60.0];
        let r = cadence_regularity(&intervals);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cadence_regularity_chaotic() {
        let intervals = vec![1.0, 1000.0, 1.0, 1000.0];
        let r = cadence_regularity(&intervals);
        assert!(r < 0.6);
    }

    #[test]
    fn sleep_pattern_all_daytime() {
        let times: Vec<DateTime<Utc>> = (8..18)
            .map(|h| Utc.with_ymd_and_hms(2026, 4, 1, h, 0, 0).unwrap())
            .collect();
        let s = sleep_pattern_presence(&times);
        assert!(
            s < 0.05,
            "no posts during sleep window → score near 0, got {s}"
        );
    }

    #[test]
    fn sleep_pattern_24x7() {
        // Posts every hour - sleep ratio = 6/24 = 0.25 → mapped to ~1.0
        let times: Vec<DateTime<Utc>> = (0..24)
            .map(|h| Utc.with_ymd_and_hms(2026, 4, 1, h, 0, 0).unwrap())
            .collect();
        let s = sleep_pattern_presence(&times);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn engagement_ratio_handles_no_data() {
        let props = serde_json::json!({});
        assert_eq!(engagement_ratio(&props), None);
    }

    #[test]
    fn engagement_ratio_capped_at_one() {
        let props = serde_json::json!({"engagement": {"likes": 1, "shares": 100}});
        assert_eq!(engagement_ratio(&props), Some(1.0));
    }

    #[test]
    fn platform_diversity_zero_when_unknown() {
        let p = platform_diversity(&serde_json::json!({}));
        assert!((p - 0.0).abs() < 1e-9);
    }

    #[test]
    fn platform_diversity_grows_with_count() {
        let one = platform_diversity(&serde_json::json!({"platform": "twitter"}));
        let many = platform_diversity(
            &serde_json::json!({"platforms": ["twitter", "telegram", "bluesky", "facebook"]}),
        );
        assert!(many > one);
    }

    #[test]
    fn fingerprint_round_trip() {
        let hg = make_hg();
        let actor = add_entity(&hg, "Alice", "test-narr");
        let mut fp = BehavioralFingerprint::empty(actor);
        fp.set_axis(BehavioralAxis::EngagementRatio, 0.42);
        store_behavioral_fingerprint(&hg, &fp).unwrap();
        let loaded = load_behavioral_fingerprint(&hg, &actor).unwrap().unwrap();
        assert_eq!(loaded.actor_id, actor);
        assert!((loaded.axis(BehavioralAxis::EngagementRatio) - 0.42).abs() < 1e-9);
        assert!(loaded
            .axis(BehavioralAxis::PostingCadenceRegularity)
            .is_nan());
    }

    #[test]
    fn compute_behavioral_runs_on_minimal_data() {
        let hg = make_hg();
        let actor = add_entity(&hg, "Alice", "test-narr");
        // Create a few situations and link the actor to them with regular intervals.
        let mut sit_ids = Vec::new();
        for h in 0..6 {
            let sit = Situation {
                id: Uuid::now_v7(),
                properties: serde_json::Value::Null,
                name: None,
                description: None,
                temporal: AllenInterval {
                    start: Some(Utc.with_ymd_and_hms(2026, 4, 1, h * 4, 0, 0).unwrap()),
                    end: None,
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
                raw_content: vec![ContentBlock::text(&format!("Post {h} #news"))],
                narrative_level: NarrativeLevel::Event,
                discourse: None,
                maturity: MaturityLevel::Candidate,
                confidence: 0.9,
                confidence_breakdown: None,
                extraction_method: ExtractionMethod::HumanEntered,
                provenance: vec![],
                narrative_id: Some("test-narr".into()),
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
            let sid = hg.create_situation(sit).unwrap();
            link(&hg, actor, sid);
            sit_ids.push(sid);
        }
        let fp = compute_behavioral_fingerprint(&hg, &actor).unwrap();
        assert_eq!(fp.sample_size, 6);
        // Regular 4-hour intervals → high regularity.
        assert!(
            fp.axis(BehavioralAxis::PostingCadenceRegularity) > 0.5,
            "expected high regularity, got {}",
            fp.axis(BehavioralAxis::PostingCadenceRegularity)
        );
        // All posts had #news exactly once each → Herfindahl = 1.0 (single hashtag dominates).
        assert!((fp.axis(BehavioralAxis::HashtagConcentration) - 1.0).abs() < 1e-9);
        // No reposts → originality = 1.0
        assert!((fp.axis(BehavioralAxis::ContentOriginality) - 1.0).abs() < 1e-9);
        // Sleep + temporal coordination + insularity remain NaN (deferred).
        assert!(fp.axis(BehavioralAxis::TemporalCoordination).is_nan());
        assert!(fp.axis(BehavioralAxis::NetworkInsularity).is_nan());
    }

    #[test]
    fn compute_disinfo_runs_on_empty_narrative() {
        let hg = make_hg();
        let fp = compute_disinfo_fingerprint(&hg, "missing-narr").unwrap();
        assert_eq!(fp.sample_size, 0);
        // All axes NaN because no situations exist.
        assert_eq!(fp.computed_axes(), 0);
    }
}
