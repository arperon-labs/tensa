//! Geocoding service using Nominatim (OpenStreetMap) with KV-backed caching.
//!
//! Converts place name strings extracted during ingestion into latitude/longitude
//! coordinates. Results are cached to avoid re-querying the same locations.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::store::KVStore;
use crate::types::SpatialPrecision;

/// KV prefix for geocode cache entries.
const GEO_PREFIX: &[u8] = b"geo/";

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
    pub async fn geocode(&self, place: &str) -> Result<Option<GeoResult>> {
        let normalized = place.trim().to_lowercase();
        if normalized.is_empty() {
            return Ok(None);
        }

        // Check cache
        let cache_key = Self::cache_key(&normalized);
        if let Some(bytes) = self.store.get(&cache_key)? {
            let entry: CacheEntry = serde_json::from_slice(&bytes)?;
            debug!(place = %normalized, cached = entry.result.is_some(), "geocode cache hit");
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
        let result = self.query_nominatim(&normalized).await;

        match result {
            Ok(geo) => {
                let entry = CacheEntry {
                    result: geo.clone(),
                };
                let bytes = serde_json::to_vec(&entry)?;
                self.store.put(&cache_key, &bytes)?;
                debug!(place = %normalized, found = geo.is_some(), "geocode result cached");
                Ok(geo)
            }
            Err(e) => {
                warn!(place = %normalized, error = %e, "geocode request failed");
                Err(e)
            }
        }
    }

    /// Query Nominatim API for a place name.
    async fn query_nominatim(&self, place: &str) -> Result<Option<GeoResult>> {
        let resp = self
            .client
            .get("https://nominatim.openstreetmap.org/search")
            .query(&[("q", place), ("format", "json"), ("limit", "1")])
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
    /// Accepts already-loaded situations to avoid N+1 re-reads.
    /// Returns the number of situations updated.
    pub async fn geocode_situations(
        &self,
        hg: &Hypergraph,
        situations: Vec<crate::types::Situation>,
    ) -> Result<usize> {
        let mut updated = 0;
        for sit in &situations {
            let description = match &sit.spatial {
                Some(sp) if sp.latitude.is_none() && sp.description.is_some() => {
                    sp.description.as_deref().unwrap()
                }
                _ => continue,
            };

            if let Some(geo) = self.geocode(description).await? {
                hg.update_situation(&sit.id, |s| {
                    if let Some(ref mut sp) = s.spatial {
                        sp.latitude = Some(geo.latitude);
                        sp.longitude = Some(geo.longitude);
                        sp.precision = geo.precision;
                    }
                })?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Batch-geocode Location entities that lack coordinates in their properties.
    /// Accepts already-loaded entities to avoid N+1 re-reads.
    /// Returns the number of entities updated.
    pub async fn geocode_location_entities(
        &self,
        hg: &Hypergraph,
        entities: Vec<crate::types::Entity>,
    ) -> Result<usize> {
        let mut updated = 0;
        for entity in &entities {
            if entity.properties.get("latitude").is_some() {
                continue;
            }
            let name = match entity.properties.get("name").and_then(|v| v.as_str()) {
                Some(n) if !n.is_empty() => n,
                _ => continue,
            };

            if let Some(geo) = self.geocode(name).await? {
                hg.update_entity_no_snapshot(&entity.id, |e| {
                    e.properties["latitude"] = serde_json::json!(geo.latitude);
                    e.properties["longitude"] = serde_json::json!(geo.longitude);
                    e.properties["geo_precision"] =
                        serde_json::to_value(&geo.precision).unwrap_or(serde_json::Value::Null);
                    e.properties["geo_display_name"] = serde_json::json!(geo.display_name);
                })?;
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Build a cache key for a normalized place name.
    fn cache_key(normalized: &str) -> Vec<u8> {
        let mut key = GEO_PREFIX.to_vec();
        key.extend_from_slice(normalized.as_bytes());
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
    }
}
