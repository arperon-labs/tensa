//! Geocoding service using Nominatim (OpenStreetMap) with KV-backed caching.
//!
//! Converts place name strings extracted during ingestion into latitude/longitude
//! coordinates. Results are cached to avoid re-querying the same locations.
//!
//! Two-stage flow:
//! 1. **Canonicalization** (optional, requires an LLM extractor): a single batch
//!    LLM call disambiguates raw place strings using narrative-setting context
//!    ("Marseilles" in 19c France → Marseille, FR — not Marseilles, IL).
//! 2. **Geocoding**: Nominatim is hit per place with an optional `&countrycodes=`
//!    filter from canonicalization. Results are cached per `(country, name)` so
//!    a wrong-country hit on one narrative doesn't poison another.
//!
//! Provenance is stamped on each Spatial / Entity update so analysts can tell
//! hard-fact coords (`Source`) from inferred ones (`LlmCanonicalized` /
//! `Geocoded`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::extraction::{NarrativeSettingHint, PlaceCanonicalization};
use crate::ingestion::llm::NarrativeExtractor;
use crate::store::KVStore;
use crate::types::{GeoProvenance, SpatialPrecision};

/// KV prefix for geocode cache entries.
const GEO_PREFIX: &[u8] = b"geo/";

/// KV prefix for canonicalization cache entries.
const GEO_CANON_PREFIX: &[u8] = b"geo/canon/";

/// Minimum interval between Nominatim requests (1 request per second per policy).
const RATE_LIMIT: Duration = Duration::from_millis(1100);

/// Geocoding result for a place name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoResult {
    pub latitude: f64,
    pub longitude: f64,
    pub precision: SpatialPrecision,
    pub display_name: String,
    pub osm_type: Option<String>,
}

/// Cached entry — `None` means "we looked this up and got no result" (avoids re-querying fictional places).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    result: Option<GeoResult>,
}

/// Nominatim API response item.
#[derive(Debug, Deserialize)]
struct NominatimResult {
    lat: String,
    lon: String,
    display_name: String,
    #[serde(rename = "type")]
    osm_type: Option<String>,
    class: Option<String>,
}

/// Geocoder with HTTP client, KV cache, and rate limiter.
pub struct Geocoder {
    client: reqwest::Client,
    store: Arc<dyn KVStore>,
    last_request: Mutex<Instant>,
}

impl Geocoder {
    /// Create a new geocoder backed by the given KV store for caching.
    pub fn new(store: Arc<dyn KVStore>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("TENSA/1.0 (narrative-engine)")
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            store,
            last_request: Mutex::new(Instant::now() - RATE_LIMIT),
        }
    }

    /// Look up a place name, returning cached result or querying Nominatim.
    /// Convenience wrapper around `geocode_with_country` with no country filter.
    pub async fn geocode(&self, place: &str) -> Result<Option<GeoResult>> {
        self.geocode_with_country(place, None).await
    }

    /// Look up a place name with an optional ISO 3166-1 alpha-2 country filter.
    /// Cache key is segregated by country so that "marseille|fr" never collides
    /// with "marseilles|us".
    pub async fn geocode_with_country(
        &self,
        place: &str,
        country_code: Option<&str>,
    ) -> Result<Option<GeoResult>> {
        let normalized = place.trim().to_lowercase();
        if normalized.is_empty() {
            return Ok(None);
        }
        let cc = country_code.map(|c| c.trim().to_lowercase()).filter(|s| !s.is_empty());

        // Check cache
        let cache_key = Self::cache_key_for(&normalized, cc.as_deref());
        if let Some(bytes) = self.store.get(&cache_key)? {
            let entry: CacheEntry = serde_json::from_slice(&bytes)?;
            debug!(place = %normalized, country = ?cc, cached = entry.result.is_some(), "geocode cache hit");
            return Ok(entry.result);
        }

        // Rate limit — release lock before sleeping to avoid blocking cached lookups
        let sleep_duration = {
            let last = self.last_request.lock().await;
            let elapsed = last.elapsed();
            if elapsed < RATE_LIMIT {
                Some(RATE_LIMIT - elapsed)
            } else {
                None
            }
        };
        if let Some(d) = sleep_duration {
            tokio::time::sleep(d).await;
        }
        {
            let mut last = self.last_request.lock().await;
            *last = Instant::now();
        }

        // Query Nominatim
        let result = self.query_nominatim(&normalized, cc.as_deref()).await;

        match result {
            Ok(geo) => {
                let entry = CacheEntry {
                    result: geo.clone(),
                };
                let bytes = serde_json::to_vec(&entry)?;
                self.store.put(&cache_key, &bytes)?;
                debug!(place = %normalized, country = ?cc, found = geo.is_some(), "geocode result cached");
                Ok(geo)
            }
            Err(e) => {
                warn!(place = %normalized, country = ?cc, error = %e, "geocode request failed");
                Err(e)
            }
        }
    }

    /// Query Nominatim API for a place name with optional country filter.
    async fn query_nominatim(
        &self,
        place: &str,
        country_code: Option<&str>,
    ) -> Result<Option<GeoResult>> {
        let mut req = self
            .client
            .get("https://nominatim.openstreetmap.org/search")
            .query(&[("q", place), ("format", "json"), ("limit", "1")]);
        if let Some(cc) = country_code {
            req = req.query(&[("countrycodes", cc)]);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| TensaError::Internal(format!("Nominatim request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(TensaError::Internal(format!(
                "Nominatim returned status {}",
                resp.status()
            )));
        }

        let results: Vec<NominatimResult> = resp
            .json()
            .await
            .map_err(|e| TensaError::Internal(format!("Nominatim parse error: {}", e)))?;

        let Some(first) = results.into_iter().next() else {
            return Ok(None);
        };

        let lat: f64 = first
            .lat
            .parse()
            .map_err(|_| TensaError::Internal("Invalid latitude from Nominatim".into()))?;
        let lon: f64 = first
            .lon
            .parse()
            .map_err(|_| TensaError::Internal("Invalid longitude from Nominatim".into()))?;

        let precision = map_precision(first.class.as_deref(), first.osm_type.as_deref());

        Ok(Some(GeoResult {
            latitude: lat,
            longitude: lon,
            precision,
            display_name: first.display_name,
            osm_type: first.osm_type,
        }))
    }

    /// Batch-geocode situations that have a spatial description but no coordinates.
    /// Backwards-compatible: no LLM canonicalization, provenance stamped `Geocoded`.
    pub async fn geocode_situations(
        &self,
        hg: &Hypergraph,
        situations: Vec<crate::types::Situation>,
    ) -> Result<usize> {
        self.geocode_situations_with_canon(hg, situations, None, None, None)
            .await
    }

    /// Batch-geocode situations with optional LLM canonicalization.
    ///
    /// If `extractor` and `setting` are both `Some`, all un-geocoded place strings
    /// are sent to the LLM in a single call to produce canonical names + country
    /// codes. Each result is then geocoded against Nominatim with the country
    /// filter, and `geo_provenance` is stamped `LlmCanonicalized`. Records that
    /// already have lat/lng (`Source` provenance) are skipped untouched.
    ///
    /// If `extractor` is `None` or canonicalization returns nothing for a given
    /// place, falls back to direct Nominatim lookup with `geo_provenance: Geocoded`.
    pub async fn geocode_situations_with_canon(
        &self,
        hg: &Hypergraph,
        situations: Vec<crate::types::Situation>,
        narrative_id: Option<&str>,
        setting: Option<&NarrativeSettingHint>,
        extractor: Option<&dyn NarrativeExtractor>,
    ) -> Result<usize> {
        // Phase 1: collect un-geocoded places + stamp Source provenance on hard-fact rows
        let mut to_resolve: Vec<(uuid::Uuid, String)> = Vec::new();
        for sit in &situations {
            match &sit.spatial {
                Some(sp) if sp.latitude.is_some() => {
                    // Hard-fact coords — stamp Source provenance if missing, then skip.
                    if sp.geo_provenance.is_none() {
                        hg.update_situation(&sit.id, |s| {
                            if let Some(ref mut sp) = s.spatial {
                                sp.geo_provenance = Some(GeoProvenance::Source);
                            }
                        })?;
                    }
                }
                Some(sp) if sp.description.is_some() => {
                    to_resolve.push((sit.id, sp.description.clone().unwrap()));
                }
                _ => {}
            }
        }
        if to_resolve.is_empty() {
            return Ok(0);
        }

        // Phase 2: optional LLM canonicalization (one call for the whole batch)
        let canon_map = self
            .canonicalize_places_batch(narrative_id, setting, extractor, &to_resolve)
            .await;

        // Phase 3: per-place Nominatim lookup with country filter from canon
        let mut updated = 0;
        for (sid, raw) in &to_resolve {
            let normalized_raw = raw.trim().to_lowercase();
            let canon = canon_map.get(&normalized_raw);
            let (query_name, country, provenance) = match canon {
                Some(c) => (
                    c.canonical_name.as_str(),
                    c.country_code.as_deref(),
                    GeoProvenance::LlmCanonicalized,
                ),
                None => (raw.as_str(), None, GeoProvenance::Geocoded),
            };

            if let Some(geo) = self.geocode_with_country(query_name, country).await? {
                hg.update_situation(sid, |s| {
                    if let Some(ref mut sp) = s.spatial {
                        sp.latitude = Some(geo.latitude);
                        sp.longitude = Some(geo.longitude);
                        sp.precision = geo.precision;
                        sp.geo_provenance = Some(provenance);
                    }
                })?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Batch-geocode Location entities. Backwards-compatible variant.
    pub async fn geocode_location_entities(
        &self,
        hg: &Hypergraph,
        entities: Vec<crate::types::Entity>,
    ) -> Result<usize> {
        self.geocode_location_entities_with_canon(hg, entities, None, None, None)
            .await
    }

    /// Batch-geocode Location entities with optional LLM canonicalization.
    /// Mirrors `geocode_situations_with_canon` — entities already carrying
    /// `latitude` in their properties are stamped `Source` and skipped.
    pub async fn geocode_location_entities_with_canon(
        &self,
        hg: &Hypergraph,
        entities: Vec<crate::types::Entity>,
        narrative_id: Option<&str>,
        setting: Option<&NarrativeSettingHint>,
        extractor: Option<&dyn NarrativeExtractor>,
    ) -> Result<usize> {
        let mut to_resolve: Vec<(uuid::Uuid, String)> = Vec::new();
        for entity in &entities {
            // A `null` value on `latitude` means a previous backfill was cleared
            // (e.g. via the cleanup route) — treat it the same as missing so the
            // next pass re-resolves. Only a non-null numeric `latitude` counts
            // as "already geocoded".
            let has_lat = matches!(
                entity.properties.get("latitude"),
                Some(v) if !v.is_null()
            );
            if has_lat {
                // Hard-fact coords — stamp Source provenance if missing or null.
                let prov_missing = matches!(
                    entity.properties.get("geo_provenance"),
                    None | Some(serde_json::Value::Null)
                );
                if prov_missing {
                    hg.update_entity_no_snapshot(&entity.id, |e| {
                        e.properties["geo_provenance"] = serde_json::json!("source");
                    })?;
                }
                continue;
            }
            let name = match entity.properties.get("name").and_then(|v| v.as_str()) {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            to_resolve.push((entity.id, name));
        }
        if to_resolve.is_empty() {
            return Ok(0);
        }

        let canon_map = self
            .canonicalize_places_batch(narrative_id, setting, extractor, &to_resolve)
            .await;

        let mut updated = 0;
        for (eid, raw) in &to_resolve {
            let normalized_raw = raw.trim().to_lowercase();
            let canon = canon_map.get(&normalized_raw);
            let (query_name, country, provenance_str) = match canon {
                Some(c) => (
                    c.canonical_name.as_str(),
                    c.country_code.as_deref(),
                    "llm_canonicalized",
                ),
                None => (raw.as_str(), None, "geocoded"),
            };

            if let Some(geo) = self.geocode_with_country(query_name, country).await? {
                let prov = provenance_str.to_string();
                hg.update_entity_no_snapshot(eid, |e| {
                    e.properties["latitude"] = serde_json::json!(geo.latitude);
                    e.properties["longitude"] = serde_json::json!(geo.longitude);
                    e.properties["geo_precision"] =
                        serde_json::to_value(&geo.precision).unwrap_or(serde_json::Value::Null);
                    e.properties["geo_display_name"] = serde_json::json!(geo.display_name);
                    e.properties["geo_provenance"] = serde_json::json!(prov);
                })?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Run the LLM batch-canonicalization pass with KV caching.
    ///
    /// Returns a `HashMap` keyed by *normalized raw place string* → canonicalization.
    /// Only places that appear in the result map have an LLM-suggested canonical
    /// name; the caller falls back to direct Nominatim for the rest.
    ///
    /// Cached at `geo/canon/{narrative_id}/{normalized_raw}` so re-runs of the
    /// same narrative skip the LLM call entirely. Without an extractor or a
    /// narrative_id, returns an empty map (no LLM call attempted).
    async fn canonicalize_places_batch(
        &self,
        narrative_id: Option<&str>,
        setting: Option<&NarrativeSettingHint>,
        extractor: Option<&dyn NarrativeExtractor>,
        places: &[(uuid::Uuid, String)],
    ) -> std::collections::HashMap<String, PlaceCanonicalization> {
        use std::collections::HashMap;
        let mut out: HashMap<String, PlaceCanonicalization> = HashMap::new();

        let Some(nid) = narrative_id else {
            return out;
        };

        // Phase A: pull from KV cache, accumulate misses
        let mut misses: Vec<(String, String)> = Vec::new();
        for (uuid, raw) in places {
            let normalized_raw = raw.trim().to_lowercase();
            if normalized_raw.is_empty() {
                continue;
            }
            let key = Self::canon_cache_key(nid, &normalized_raw);
            match self.store.get(&key) {
                Ok(Some(bytes)) => {
                    if let Ok(canon) = serde_json::from_slice::<PlaceCanonicalization>(&bytes) {
                        out.insert(normalized_raw, canon);
                        continue;
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(narrative_id = %nid, raw = %normalized_raw, error = %e, "canon cache read failed");
                }
            }
            misses.push((uuid.to_string(), raw.clone()));
        }

        // Phase B: if we have any misses AND an extractor + setting, batch-call the LLM.
        let (Some(extractor), Some(setting)) = (extractor, setting) else {
            return out;
        };
        if misses.is_empty() {
            return out;
        }

        let rows = match extractor.canonicalize_places(setting, &misses) {
            Ok(r) => r,
            Err(e) => {
                warn!(narrative_id = %nid, error = %e, "canonicalize_places LLM call failed");
                return out;
            }
        };

        // Phase C: write through to cache + accumulate
        for canon in rows {
            let normalized_raw = canon.raw_name.trim().to_lowercase();
            if normalized_raw.is_empty() {
                continue;
            }
            let key = Self::canon_cache_key(nid, &normalized_raw);
            if let Ok(bytes) = serde_json::to_vec(&canon) {
                let _ = self.store.put(&key, &bytes);
            }
            out.insert(normalized_raw, canon);
        }
        out
    }

    /// Diagnostic helper: exposes `canonicalize_places_batch` for routes that need
    /// to inspect what the geocoder's batch step actually produces (vs calling
    /// the trait method directly through the extractor).
    pub async fn debug_canonicalize_places_batch(
        &self,
        narrative_id: &str,
        setting: &NarrativeSettingHint,
        extractor: &dyn NarrativeExtractor,
        places: &[(uuid::Uuid, String)],
    ) -> std::collections::HashMap<String, PlaceCanonicalization> {
        self.canonicalize_places_batch(Some(narrative_id), Some(setting), Some(extractor), places)
            .await
    }

    /// Build a cache key for a normalized place name (no country filter).
    fn cache_key(normalized: &str) -> Vec<u8> {
        Self::cache_key_for(normalized, None)
    }

    /// Build a cache key for a normalized place name with optional country filter.
    /// Country-scoped keys are stored as `geo/{cc}|{name}` so that the same raw
    /// name resolved against a different country never collides.
    fn cache_key_for(normalized: &str, country_code: Option<&str>) -> Vec<u8> {
        let mut key = GEO_PREFIX.to_vec();
        if let Some(cc) = country_code {
            key.extend_from_slice(cc.as_bytes());
            key.push(b'|');
        }
        key.extend_from_slice(normalized.as_bytes());
        key
    }

    /// Build a cache key for a canonicalization entry.
    fn canon_cache_key(narrative_id: &str, normalized_raw: &str) -> Vec<u8> {
        let mut key = GEO_CANON_PREFIX.to_vec();
        key.extend_from_slice(narrative_id.as_bytes());
        key.push(b'/');
        key.extend_from_slice(normalized_raw.as_bytes());
        key
    }
}

/// Map Nominatim class/type to SpatialPrecision.
fn map_precision(class: Option<&str>, osm_type: Option<&str>) -> SpatialPrecision {
    match (class, osm_type) {
        (Some("building"), _) | (Some("amenity"), _) | (_, Some("house")) => {
            SpatialPrecision::Exact
        }
        (Some("place"), Some(t))
            if matches!(
                t,
                "suburb" | "city_district" | "neighbourhood" | "quarter" | "borough"
            ) =>
        {
            SpatialPrecision::Area
        }
        (Some("place"), Some(t))
            if matches!(
                t,
                "city" | "town" | "village" | "hamlet" | "county" | "state" | "municipality"
            ) =>
        {
            SpatialPrecision::Region
        }
        (Some("boundary"), _) => SpatialPrecision::Region,
        _ => SpatialPrecision::Approximate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;

    fn test_store() -> Arc<dyn KVStore> {
        Arc::new(MemoryStore::new())
    }

    #[test]
    fn test_cache_key_format() {
        let key = Geocoder::cache_key("new york");
        assert_eq!(&key[..4], b"geo/");
        assert_eq!(&key[4..], b"new york");
    }

    #[test]
    fn test_map_precision_building() {
        assert_eq!(
            map_precision(Some("building"), Some("yes")),
            SpatialPrecision::Exact
        );
        assert_eq!(
            map_precision(Some("amenity"), Some("restaurant")),
            SpatialPrecision::Exact
        );
    }

    #[test]
    fn test_map_precision_area() {
        assert_eq!(
            map_precision(Some("place"), Some("suburb")),
            SpatialPrecision::Area
        );
        assert_eq!(
            map_precision(Some("place"), Some("neighbourhood")),
            SpatialPrecision::Area
        );
    }

    #[test]
    fn test_map_precision_region() {
        assert_eq!(
            map_precision(Some("place"), Some("city")),
            SpatialPrecision::Region
        );
        assert_eq!(
            map_precision(Some("boundary"), Some("administrative")),
            SpatialPrecision::Region
        );
    }

    #[test]
    fn test_map_precision_approximate() {
        assert_eq!(
            map_precision(Some("natural"), Some("peak")),
            SpatialPrecision::Approximate
        );
        assert_eq!(map_precision(None, None), SpatialPrecision::Approximate);
    }

    #[tokio::test]
    async fn test_geocode_empty_place() {
        let geocoder = Geocoder::new(test_store());
        let result = geocoder.geocode("").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_geocode_cache_roundtrip() {
        let store = test_store();
        // Pre-populate cache
        let entry = CacheEntry {
            result: Some(GeoResult {
                latitude: 48.8566,
                longitude: 2.3522,
                precision: SpatialPrecision::Region,
                display_name: "Paris, France".into(),
                osm_type: Some("city".into()),
            }),
        };
        let key = Geocoder::cache_key("paris");
        store
            .put(&key, &serde_json::to_vec(&entry).unwrap())
            .unwrap();

        let geocoder = Geocoder::new(store);
        let result = geocoder.geocode("Paris").await.unwrap();
        assert!(result.is_some());
        let geo = result.unwrap();
        assert!((geo.latitude - 48.8566).abs() < 0.001);
        assert!((geo.longitude - 2.3522).abs() < 0.001);
        assert_eq!(geo.precision, SpatialPrecision::Region);
    }

    #[tokio::test]
    async fn test_geocode_cache_negative() {
        let store = test_store();
        // Cache a negative result (fictional place)
        let entry = CacheEntry { result: None };
        let key = Geocoder::cache_key("mordor");
        store
            .put(&key, &serde_json::to_vec(&entry).unwrap())
            .unwrap();

        let geocoder = Geocoder::new(store);
        let result = geocoder.geocode("Mordor").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_geocode_situations_updates_spatial() {
        use crate::types::*;
        use chrono::Utc;
        use uuid::Uuid;

        let store = test_store();

        // Pre-populate geocode cache for "St Petersburg"
        let entry = CacheEntry {
            result: Some(GeoResult {
                latitude: 59.9343,
                longitude: 30.3351,
                precision: SpatialPrecision::Region,
                display_name: "Saint Petersburg, Russia".into(),
                osm_type: Some("city".into()),
            }),
        };
        let cache_key = Geocoder::cache_key("st petersburg");
        store
            .put(&cache_key, &serde_json::to_vec(&entry).unwrap())
            .unwrap();

        let hg = Hypergraph::new(store.clone());
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
            spatial: Some(SpatialAnchor {
                latitude: None,
                longitude: None,
                precision: SpatialPrecision::Unknown,
                location_entity: None,
                location_name: Some("St Petersburg".into()),
                description: Some("St Petersburg".into()),
                geo_provenance: None,
            }),
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Event,
            narrative_id: None,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
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
            source_chunk_id: None,
            source_span: None,
        };
        let sid = hg.create_situation(sit).unwrap();

        let geocoder = Geocoder::new(store);
        let sit_to_geocode = hg.get_situation(&sid).unwrap();
        let count = geocoder
            .geocode_situations(&hg, vec![sit_to_geocode])
            .await
            .unwrap();
        assert_eq!(count, 1);

        let updated = hg.get_situation(&sid).unwrap();
        let sp = updated.spatial.unwrap();
        assert!((sp.latitude.unwrap() - 59.9343).abs() < 0.001);
        assert!((sp.longitude.unwrap() - 30.3351).abs() < 0.001);
        assert_eq!(sp.precision, SpatialPrecision::Region);
        // Provenance is `Geocoded` on the no-extractor fallback path.
        assert_eq!(sp.geo_provenance, Some(GeoProvenance::Geocoded));
    }

    #[test]
    fn test_country_scoped_cache_keys_dont_collide() {
        let no_country = Geocoder::cache_key_for("marseille", None);
        let france = Geocoder::cache_key_for("marseille", Some("fr"));
        let illinois = Geocoder::cache_key_for("marseilles", Some("us"));
        assert_ne!(no_country, france);
        assert_ne!(france, illinois);
        // Only the country-scoped variants carry the `|` separator.
        assert!(france.windows(1).any(|w| w == b"|"));
        assert!(!no_country.contains(&b'|'));
    }

    #[tokio::test]
    async fn test_hard_fact_coords_skip_geocoding_and_stamp_source() {
        use crate::types::*;
        use chrono::Utc;
        use uuid::Uuid;

        let store = test_store();
        let hg = Hypergraph::new(store.clone());
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
            spatial: Some(SpatialAnchor {
                // Hard-fact coords from the source — provenance not yet stamped.
                latitude: Some(43.2965),
                longitude: Some(5.3698),
                precision: SpatialPrecision::Region,
                location_entity: None,
                location_name: Some("Marseille".into()),
                description: Some("Marseille".into()),
                geo_provenance: None,
            }),
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Event,
            narrative_id: None,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::StructuredImport,
            provenance: vec![],
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
            source_chunk_id: None,
            source_span: None,
        };
        let sid = hg.create_situation(sit).unwrap();

        let geocoder = Geocoder::new(store);
        let sit_loaded = hg.get_situation(&sid).unwrap();
        let count = geocoder
            .geocode_situations(&hg, vec![sit_loaded])
            .await
            .unwrap();
        // Hard-fact coords were untouched — `updated` should not increment.
        assert_eq!(count, 0);

        let after = hg.get_situation(&sid).unwrap();
        let sp = after.spatial.unwrap();
        // Coords preserved verbatim.
        assert!((sp.latitude.unwrap() - 43.2965).abs() < 1e-6);
        assert!((sp.longitude.unwrap() - 5.3698).abs() < 1e-6);
        // Provenance now stamped Source.
        assert_eq!(sp.geo_provenance, Some(GeoProvenance::Source));
    }

    #[tokio::test]
    async fn test_canonicalization_routes_to_country_scoped_cache() {
        use crate::ingestion::extraction::PlaceCanonicalization;
        use crate::types::*;
        use chrono::Utc;
        use uuid::Uuid;

        let store = test_store();
        let hg = Hypergraph::new(store.clone());

        // Pre-populate canonicalization cache so no LLM call is required.
        let canon = PlaceCanonicalization {
            uuid: "ignored".into(),
            raw_name: "Marseilles".into(),
            canonical_name: "Marseille".into(),
            country_code: Some("fr".into()),
            admin_region: None,
            confidence: 0.95,
        };
        let canon_key = Geocoder::canon_cache_key("monte-cristo", "marseilles");
        store
            .put(&canon_key, &serde_json::to_vec(&canon).unwrap())
            .unwrap();

        // Pre-populate the country-scoped Nominatim cache so no HTTP call happens.
        let entry = CacheEntry {
            result: Some(GeoResult {
                latitude: 43.2965,
                longitude: 5.3698,
                precision: SpatialPrecision::Region,
                display_name: "Marseille, France".into(),
                osm_type: Some("city".into()),
            }),
        };
        let geo_key = Geocoder::cache_key_for("marseille", Some("fr"));
        store
            .put(&geo_key, &serde_json::to_vec(&entry).unwrap())
            .unwrap();

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
            spatial: Some(SpatialAnchor {
                latitude: None,
                longitude: None,
                precision: SpatialPrecision::Unknown,
                location_entity: None,
                location_name: Some("Marseilles".into()),
                description: Some("Marseilles".into()),
                geo_provenance: None,
            }),
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![],
            narrative_level: NarrativeLevel::Event,
            narrative_id: Some("monte-cristo".into()),
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
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
            source_chunk_id: None,
            source_span: None,
        };
        let sid = hg.create_situation(sit).unwrap();

        // Setting hint is supplied but no extractor — the canon cache hit alone
        // should be enough to route through the country-scoped Nominatim cache.
        let setting = NarrativeSettingHint {
            setting: "Early-19c France/Italy".into(),
            country_hint: Some("fr".into()),
        };

        let geocoder = Geocoder::new(store);
        let sit_loaded = hg.get_situation(&sid).unwrap();
        let count = geocoder
            .geocode_situations_with_canon(
                &hg,
                vec![sit_loaded],
                Some("monte-cristo"),
                Some(&setting),
                None,
            )
            .await
            .unwrap();
        assert_eq!(count, 1);

        let after = hg.get_situation(&sid).unwrap();
        let sp = after.spatial.unwrap();
        assert!((sp.latitude.unwrap() - 43.2965).abs() < 1e-4);
        assert!((sp.longitude.unwrap() - 5.3698).abs() < 1e-4);
        assert_eq!(sp.geo_provenance, Some(GeoProvenance::LlmCanonicalized));
    }
}
