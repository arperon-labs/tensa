//! Spread velocity monitoring + anomaly detection (Sprint D2).
//!
//! Maintains a per-platform R₀ baseline (mean + std-dev over a sliding window
//! of historical spread observations). When a new R₀ measurement comes in,
//! the monitor computes its z-score against the baseline; values beyond
//! `ANOMALY_Z_THRESHOLD` (default 2σ) fire a [`VelocityAlert`] persisted at
//! `vm/alert/{ts_be}/{narrative_id}`.
//!
//! Operational target (per spec §2.1): the "2-hour window before a false
//! story goes fully viral" is the prebunking opportunity. A 2σ alert on a
//! Twitter-like β ≈ 0.45 baseline trips well inside that window.
//!
//! The synthetic-baseline / learned-baseline distinction promised by Sprint
//! D7.5 is **stubbed** here (default `BaselineModel::synthetic_for(platform)`
//! is used until the per-platform sample count crosses
//! [`MIN_LEARNED_SAMPLES`] = 100); the surfaced `BaselineSource` enum lets
//! analysts tell the difference. Full D7.5 wiring lands post-D6.

#![cfg(feature = "disinfo")]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, TensaError};
use crate::hypergraph::{keys, Hypergraph};
use crate::types::Platform;

/// Z-score threshold above which an observation is flagged as anomalous.
pub const ANOMALY_Z_THRESHOLD: f64 = 2.0;

/// Number of samples required before a baseline is treated as "learned"
/// rather than "synthetic".
pub const MIN_LEARNED_SAMPLES: usize = 100;

/// Whether a baseline is filled with research-derived synthetic defaults
/// or was learned from observed spread data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BaselineSource {
    Synthetic,
    Learned,
}

/// Sliding-window R₀ baseline for a single platform / narrative-kind bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineModel {
    pub platform: String,
    /// Optional narrative-kind tag for bucketing (e.g. "political", "celebrity").
    pub narrative_kind: String,
    pub mean_r0: f64,
    pub std_r0: f64,
    pub sample_count: usize,
    pub source: BaselineSource,
    pub updated_at: DateTime<Utc>,
}

impl BaselineModel {
    /// Synthetic baseline tuned to published organic R₀ ranges per platform
    /// (Sprint D7.5 will reload these from a config; for now they're inline).
    pub fn synthetic_for(platform: &Platform, narrative_kind: &str) -> Self {
        let (mean, std) = match platform {
            Platform::Twitter => (1.6, 0.4),
            Platform::Telegram => (1.1, 0.3),
            Platform::Bluesky => (0.9, 0.25),
            Platform::Reddit => (1.2, 0.35),
            Platform::Facebook => (1.4, 0.4),
            Platform::Instagram => (0.8, 0.25),
            Platform::TikTok => (1.8, 0.5),
            Platform::YouTube => (0.7, 0.2),
            Platform::Mastodon => (0.6, 0.2),
            Platform::VKontakte => (1.0, 0.3),
            Platform::Rss | Platform::Web => (0.5, 0.2),
            Platform::Other(_) => (1.0, 0.3),
        };
        Self {
            platform: platform.as_index_str().to_string(),
            narrative_kind: narrative_kind.to_string(),
            mean_r0: mean,
            std_r0: std,
            sample_count: 0,
            source: BaselineSource::Synthetic,
            updated_at: Utc::now(),
        }
    }

    /// Update with a new observation. Welford-style online mean+variance —
    /// numerically stable, no need to keep the full sample. Promotes the
    /// source from `Synthetic` to `Learned` once the threshold is crossed.
    pub fn observe(&mut self, r0: f64) {
        if !r0.is_finite() {
            return;
        }
        // Drop the synthetic prior **only on the very first observation** —
        // checking sample_count == 0 ensures the reset doesn't keep firing
        // and prevents `sample_count` from getting stuck below the learned
        // promotion threshold.
        if self.source == BaselineSource::Synthetic && self.sample_count == 0 {
            self.mean_r0 = 0.0;
            self.std_r0 = 0.0;
        }
        let n = self.sample_count + 1;
        let delta = r0 - self.mean_r0;
        let new_mean = self.mean_r0 + delta / n as f64;
        let m2 = self.std_r0.powi(2) * self.sample_count as f64 + delta * (r0 - new_mean);
        self.mean_r0 = new_mean;
        self.std_r0 = (m2 / n as f64).sqrt();
        self.sample_count = n;
        self.updated_at = Utc::now();
        if self.sample_count >= MIN_LEARNED_SAMPLES {
            self.source = BaselineSource::Learned;
        }
    }

    /// Z-score of `r0` against this baseline. Returns 0.0 when std is zero
    /// (no spread to detect against).
    pub fn z_score(&self, r0: f64) -> f64 {
        if self.std_r0 < 1e-9 {
            0.0
        } else {
            (r0 - self.mean_r0) / self.std_r0
        }
    }
}

/// Velocity alert fired when an observed R₀ blows past the baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityAlert {
    pub id: Uuid,
    pub narrative_id: String,
    pub platform: String,
    pub narrative_kind: String,
    pub observed_r0: f64,
    pub baseline_mean: f64,
    pub baseline_std: f64,
    pub baseline_source: BaselineSource,
    pub baseline_sample_count: usize,
    pub z_score: f64,
    pub fired_at: DateTime<Utc>,
}

/// Velocity monitor — front-end to the per-platform baselines + alert log.
///
/// Stateless wrapper around the KV store: baselines persist at
/// [`keys::VELOCITY_BASELINE`], alerts at [`keys::VELOCITY_ALERT`].
pub struct VelocityMonitor<'a> {
    hypergraph: &'a Hypergraph,
}

impl<'a> VelocityMonitor<'a> {
    pub fn new(hypergraph: &'a Hypergraph) -> Self {
        Self { hypergraph }
    }

    fn baseline_key(platform: &Platform, narrative_kind: &str) -> Vec<u8> {
        let mut key = keys::VELOCITY_BASELINE.to_vec();
        key.extend_from_slice(platform.as_index_str().as_bytes());
        key.push(b'/');
        key.extend_from_slice(narrative_kind.as_bytes());
        key
    }

    fn alert_key(ts: DateTime<Utc>, narrative_id: &str) -> Vec<u8> {
        // Prefix on narrative_id first so `recent_alerts` can do a selective
        // prefix scan instead of reading every alert across all narratives.
        let mut key = keys::VELOCITY_ALERT.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(&ts.timestamp_millis().to_be_bytes());
        key
    }

    fn alert_prefix(narrative_id: &str) -> Vec<u8> {
        let mut prefix = keys::VELOCITY_ALERT.to_vec();
        prefix.extend_from_slice(narrative_id.as_bytes());
        prefix.push(b'/');
        prefix
    }

    /// Load (or synthesize) the baseline for a given platform / narrative-kind.
    pub fn baseline(&self, platform: &Platform, narrative_kind: &str) -> Result<BaselineModel> {
        let key = Self::baseline_key(platform, narrative_kind);
        match self.hypergraph.store().get(&key)? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)
                .map_err(|e| TensaError::Serialization(e.to_string()))?),
            None => Ok(BaselineModel::synthetic_for(platform, narrative_kind)),
        }
    }

    fn put_baseline(&self, baseline: &BaselineModel, platform: &Platform) -> Result<()> {
        let key = Self::baseline_key(platform, &baseline.narrative_kind);
        let bytes =
            serde_json::to_vec(baseline).map_err(|e| TensaError::Serialization(e.to_string()))?;
        self.hypergraph.store().put(&key, &bytes)
    }

    /// Record a new R₀ observation. Returns a [`VelocityAlert`] when the
    /// z-score exceeds [`ANOMALY_Z_THRESHOLD`]; otherwise updates the baseline
    /// and returns `None`.
    ///
    /// This is the single integration point for D2's per-platform R₀ output:
    /// after `run_smir_contagion` produces `r0_by_platform`, callers iterate
    /// the map and feed each `(platform, r0)` pair through `check_anomaly`.
    pub fn check_anomaly(
        &self,
        narrative_id: &str,
        platform: &Platform,
        narrative_kind: &str,
        observed_r0: f64,
    ) -> Result<Option<VelocityAlert>> {
        let mut baseline = self.baseline(platform, narrative_kind)?;
        let z = baseline.z_score(observed_r0);
        let alert = if z.abs() >= ANOMALY_Z_THRESHOLD {
            let alert = VelocityAlert {
                id: Uuid::now_v7(),
                narrative_id: narrative_id.to_string(),
                platform: platform.as_index_str().to_string(),
                narrative_kind: narrative_kind.to_string(),
                observed_r0,
                baseline_mean: baseline.mean_r0,
                baseline_std: baseline.std_r0,
                baseline_source: baseline.source,
                baseline_sample_count: baseline.sample_count,
                z_score: z,
                fired_at: Utc::now(),
            };
            let key = Self::alert_key(alert.fired_at, narrative_id);
            self.hypergraph.store().put(
                &key,
                &serde_json::to_vec(&alert)
                    .map_err(|e| TensaError::Serialization(e.to_string()))?,
            )?;
            Some(alert)
        } else {
            None
        };
        // Always observe — alerts inform the analyst, but the baseline must
        // still drift toward reality.
        baseline.observe(observed_r0);
        self.put_baseline(&baseline, platform)?;
        Ok(alert)
    }

    /// List recent alerts for a narrative (most recent first). Selective scan
    /// keyed on `vm/alert/{narrative_id}/` so cost scales with the narrative's
    /// own alert volume, not the global alert log.
    pub fn recent_alerts(&self, narrative_id: &str, limit: usize) -> Result<Vec<VelocityAlert>> {
        let prefix = Self::alert_prefix(narrative_id);
        let pairs = self.hypergraph.store().prefix_scan(&prefix)?;
        let mut alerts: Vec<VelocityAlert> = Vec::with_capacity(pairs.len());
        for (_, value) in pairs {
            if let Ok(alert) = serde_json::from_slice::<VelocityAlert>(&value) {
                alerts.push(alert);
            }
        }
        alerts.sort_by(|a, b| b.fired_at.cmp(&a.fired_at));
        alerts.truncate(limit);
        Ok(alerts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use std::sync::Arc;

    fn make_hg() -> Hypergraph {
        Hypergraph::new(Arc::new(MemoryStore::new()))
    }

    #[test]
    fn synthetic_baseline_per_platform() {
        let twitter = BaselineModel::synthetic_for(&Platform::Twitter, "political");
        let bluesky = BaselineModel::synthetic_for(&Platform::Bluesky, "political");
        assert!(
            twitter.mean_r0 > bluesky.mean_r0,
            "Twitter R₀ should outpace Bluesky in synthetic defaults"
        );
        assert_eq!(twitter.source, BaselineSource::Synthetic);
        assert_eq!(twitter.sample_count, 0);
    }

    #[test]
    fn z_score_returns_zero_with_zero_std() {
        let mut baseline = BaselineModel::synthetic_for(&Platform::Twitter, "x");
        baseline.std_r0 = 0.0;
        assert_eq!(baseline.z_score(5.0), 0.0);
    }

    #[test]
    fn observe_promotes_to_learned_after_threshold() {
        let mut baseline = BaselineModel::synthetic_for(&Platform::Twitter, "x");
        for _ in 0..MIN_LEARNED_SAMPLES {
            baseline.observe(1.5);
        }
        assert_eq!(baseline.source, BaselineSource::Learned);
        assert!(baseline.sample_count >= MIN_LEARNED_SAMPLES);
    }

    #[test]
    fn check_anomaly_fires_on_2sigma_spike() {
        let hg = make_hg();
        let monitor = VelocityMonitor::new(&hg);
        // Twitter synthetic mean ≈ 1.6, std ≈ 0.4 → 4.0 is z ≈ 6, well past 2σ.
        let alert = monitor
            .check_anomaly("narr-x", &Platform::Twitter, "political", 4.0)
            .unwrap();
        let alert = alert.expect("expected alert at z>2");
        assert!(alert.z_score >= ANOMALY_Z_THRESHOLD);
        assert_eq!(alert.platform, "twitter");
        assert_eq!(alert.baseline_source, BaselineSource::Synthetic);
    }

    #[test]
    fn check_anomaly_silent_within_baseline() {
        let hg = make_hg();
        let monitor = VelocityMonitor::new(&hg);
        // Twitter synthetic mean ≈ 1.6 — observation right on the mean is silent.
        let alert = monitor
            .check_anomaly("narr-x", &Platform::Twitter, "political", 1.6)
            .unwrap();
        assert!(alert.is_none());
    }

    #[test]
    fn alerts_round_trip_per_narrative() {
        let hg = make_hg();
        let monitor = VelocityMonitor::new(&hg);
        // Two different narrative_kinds → independent baselines, so each
        // alert fires against the synthetic prior instead of the second
        // observation seeing a baseline already shifted by the first.
        let _ = monitor
            .check_anomaly("narr-a", &Platform::Twitter, "kind-a", 5.0)
            .unwrap();
        let _ = monitor
            .check_anomaly("narr-b", &Platform::Twitter, "kind-b", 5.5)
            .unwrap();
        let a = monitor.recent_alerts("narr-a", 10).unwrap();
        let b = monitor.recent_alerts("narr-b", 10).unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].narrative_id, "narr-a");
        assert_eq!(b[0].narrative_id, "narr-b");
    }

    #[test]
    fn velocity_baseline_full_lifecycle_d7_5() {
        // Verifies D7.5 requirements:
        // 1. VelocityMonitor::new() initializes with synthetic baselines
        // 2. BaselineSource enum has Synthetic and Learned variants with sample_count
        // 3. Auto-promotes from Synthetic to Learned at 100 samples

        let hg = make_hg();
        let monitor = VelocityMonitor::new(&hg);

        // Step 1: Fresh baseline is Synthetic with sample_count=0
        let baseline = monitor.baseline(&Platform::Twitter, "test-kind").unwrap();
        assert_eq!(baseline.source, BaselineSource::Synthetic);
        assert_eq!(baseline.sample_count, 0);
        assert!(baseline.mean_r0 > 0.0, "synthetic prior should be non-zero");
        assert!(baseline.std_r0 > 0.0, "synthetic std should be non-zero");

        // Step 2: Feed observations. After the first observation the synthetic
        // prior is reset, so we accumulate from there.
        for _i in 0..50 {
            let _ = monitor
                .check_anomaly("narr-lifecycle", &Platform::Twitter, "test-kind", 1.5)
                .unwrap();
        }
        let mid = monitor.baseline(&Platform::Twitter, "test-kind").unwrap();
        assert_eq!(mid.source, BaselineSource::Synthetic);
        assert_eq!(mid.sample_count, 50);

        // Step 3: Continue until the 100-sample threshold
        for _ in 50..MIN_LEARNED_SAMPLES {
            let _ = monitor
                .check_anomaly("narr-lifecycle", &Platform::Twitter, "test-kind", 1.5)
                .unwrap();
        }
        let learned = monitor.baseline(&Platform::Twitter, "test-kind").unwrap();
        assert_eq!(learned.source, BaselineSource::Learned);
        assert!(learned.sample_count >= MIN_LEARNED_SAMPLES);
        // Mean should have converged toward 1.5
        assert!(
            (learned.mean_r0 - 1.5).abs() < 0.05,
            "mean should converge to observed value; got {}",
            learned.mean_r0
        );
    }
}
