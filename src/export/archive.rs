//! Tensa Narrative Archive (.tensa) — export logic.
//!
//! Creates a ZIP archive containing JSON files organized by narrative,
//! supporting lossless round-trip export/import between Tensa instances.

use std::collections::HashSet;
use std::io::{Cursor, Write};

use chrono::Utc;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

use crate::analysis::community;
use crate::error::{Result, TensaError};
use crate::hypergraph::Hypergraph;
use crate::ingestion::prompt_tuning;
use crate::narrative::project::ProjectRegistry;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::taxonomy::TaxonomyRegistry;
use crate::narrative::types::TaxonomyEntry;

use super::archive_types::*;
use super::collect_narrative_data;

/// Export one or more narratives as a `.tensa` ZIP archive.
pub fn export_archive(
    narrative_ids: &[&str],
    hypergraph: &Hypergraph,
    opts: &ArchiveExportOptions,
) -> Result<Vec<u8>> {
    let buf = Vec::new();
    let cursor = Cursor::new(buf);
    let mut zip = ZipWriter::new(cursor);
    let file_opts =
        SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    let store = hypergraph.store_arc();
    let nar_registry = NarrativeRegistry::new(store.clone());

    // Determine which layers are actually populated
    let mut layers = ArchiveLayers::default();

    // Collect all referenced project IDs for later export
    let mut project_ids: HashSet<String> = HashSet::new();

    // ── Per-narrative export ──────────────────────────────────
    for &nar_id in narrative_ids {
        let prefix = format!("narratives/{}/", nar_id);

        // narrative.json (required)
        let narrative = nar_registry.get(nar_id)?;
        if let Some(ref pid) = narrative.project_id {
            project_ids.insert(pid.clone());
        }
        write_json(
            &mut zip,
            &file_opts,
            &format!("{prefix}narrative.json"),
            &narrative,
            opts.pretty,
        )?;

        // Core graph data — destructure to take ownership
        let mut core = collect_narrative_data(nar_id, hypergraph)?;

        // EATH Phase 3: strip synthetic records by default. The
        // `include_synthetic` opt-in propagates from the API layer through
        // ArchiveExportOptions; once filtered we also drop participations
        // whose endpoints are gone (avoids dangling refs in the archive).
        if !opts.include_synthetic {
            core.entities = crate::synth::emit::filter_synthetic_entities(core.entities, false);
            core.situations =
                crate::synth::emit::filter_synthetic_situations(core.situations, false);
            let kept_e: HashSet<_> = core.entities.iter().map(|e| e.id).collect();
            let kept_s: HashSet<_> = core.situations.iter().map(|s| s.id).collect();
            core.participations.retain(|p| {
                kept_e.contains(&p.entity_id)
                    && kept_s.contains(&p.situation_id)
                    && !crate::synth::emit::is_synthetic_participation(p)
            });
            core.causal_links
                .retain(|l| kept_s.contains(&l.from_situation) && kept_s.contains(&l.to_situation));
        }

        let entity_ids: HashSet<_> = core.entities.iter().map(|e| e.id).collect();
        let situation_ids: HashSet<_> = core.situations.iter().map(|s| s.id).collect();

        // Save embeddings before potentially stripping them
        let entity_embeddings_saved: Vec<_> = if opts.include_embeddings {
            core.entities
                .iter()
                .filter_map(|e| e.embedding.as_ref().map(|emb| (e.id, emb.clone())))
                .collect()
        } else {
            Vec::new()
        };
        let sit_embeddings_saved: Vec<_> = if opts.include_embeddings {
            core.situations
                .iter()
                .filter_map(|s| s.embedding.as_ref().map(|emb| (s.id, emb.clone())))
                .collect()
        } else {
            Vec::new()
        };

        // Move out of core, strip embeddings in-place (no clone)
        let super::NarrativeExport {
            entities,
            situations,
            participations,
            causal_links,
            ..
        } = core;
        let mut entities = entities;
        let mut situations = situations;
        if !opts.include_embeddings {
            for e in &mut entities {
                e.embedding = None;
            }
            for s in &mut situations {
                s.embedding = None;
            }
        }

        write_json(
            &mut zip,
            &file_opts,
            &format!("{prefix}entities.json"),
            &entities,
            opts.pretty,
        )?;
        write_json(
            &mut zip,
            &file_opts,
            &format!("{prefix}situations.json"),
            &situations,
            opts.pretty,
        )?;
        write_json(
            &mut zip,
            &file_opts,
            &format!("{prefix}participations.json"),
            &participations,
            opts.pretty,
        )?;

        // Causal links (optional, but always written if present)
        if !causal_links.is_empty() {
            write_json(
                &mut zip,
                &file_opts,
                &format!("{prefix}causal_links.json"),
                &causal_links,
                opts.pretty,
            )?;
        }

        // ── Sources layer ────────────────────────────────────
        if opts.include_sources {
            let mut sources = Vec::new();
            let mut attributions = Vec::new();
            let mut contentions = Vec::new();
            let mut source_ids: HashSet<uuid::Uuid> = HashSet::new();
            let mut seen_contentions: HashSet<(uuid::Uuid, uuid::Uuid)> = HashSet::new();

            // Collect attributions for all entities and situations in this narrative
            for eid in &entity_ids {
                if let Ok(attrs) = hypergraph.get_attributions_for_target(eid) {
                    for attr in attrs {
                        source_ids.insert(attr.source_id);
                        attributions.push(attr);
                    }
                }
            }
            for sid in &situation_ids {
                if let Ok(attrs) = hypergraph.get_attributions_for_target(sid) {
                    for attr in attrs {
                        source_ids.insert(attr.source_id);
                        attributions.push(attr);
                    }
                }
                // Contentions for situations in this narrative (dedup via canonical pair)
                if let Ok(links) = hypergraph.get_contentions_for_situation(sid) {
                    for link in links {
                        if situation_ids.contains(&link.situation_a)
                            && situation_ids.contains(&link.situation_b)
                        {
                            let pair = if link.situation_a < link.situation_b {
                                (link.situation_a, link.situation_b)
                            } else {
                                (link.situation_b, link.situation_a)
                            };
                            if seen_contentions.insert(pair) {
                                contentions.push(link);
                            }
                        }
                    }
                }
            }

            // Resolve referenced sources
            for src_id in &source_ids {
                if let Ok(src) = hypergraph.get_source(src_id) {
                    sources.push(src);
                }
            }

            if !sources.is_empty() || !attributions.is_empty() || !contentions.is_empty() {
                layers.sources = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}sources/sources.json"),
                    &sources,
                    opts.pretty,
                )?;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}sources/attributions.json"),
                    &attributions,
                    opts.pretty,
                )?;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}sources/contentions.json"),
                    &contentions,
                    opts.pretty,
                )?;
            }
        }

        // ── Chunks layer ─────────────────────────────────────
        if opts.include_chunks {
            let mut chunks = hypergraph.list_chunks_by_narrative(nar_id)?;
            if !chunks.is_empty() {
                layers.chunks = true;
                if !opts.include_embeddings {
                    for c in &mut chunks {
                        c.embedding = None;
                    }
                }
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}chunks/chunks.json"),
                    &chunks,
                    opts.pretty,
                )?;
            }
        }

        // ── State versions layer ─────────────────────────────
        if opts.include_state_versions {
            let mut state_versions = Vec::new();
            for eid in &entity_ids {
                if let Ok(versions) = hypergraph.get_state_history(eid) {
                    state_versions.extend(versions);
                }
            }
            if !state_versions.is_empty() {
                layers.state_versions = true;
                if !opts.include_embeddings {
                    for sv in &mut state_versions {
                        sv.embedding = None;
                    }
                }
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}state_versions/state_versions.json"),
                    &state_versions,
                    opts.pretty,
                )?;
            }
        }

        // ── Inference results layer ──────────────────────────
        if opts.include_inference {
            // Scan all inference results and filter to those targeting our entities/situations
            let all_results = scan_inference_results(&*store)?;
            let nar_results: Vec<_> = all_results
                .into_iter()
                .filter(|r| {
                    entity_ids.contains(&r.target_id) || situation_ids.contains(&r.target_id)
                })
                .collect();
            if !nar_results.is_empty() {
                layers.inference = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}inference/results.json"),
                    &nar_results,
                    opts.pretty,
                )?;
            }
        }

        // ── Analysis layer ───────────────────────────────────
        if opts.include_analysis {
            let communities = community::list_summaries(&*store, nar_id)?;
            if !communities.is_empty() {
                layers.analysis = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}analysis/communities.json"),
                    &communities,
                    opts.pretty,
                )?;
            }
        }

        // ── Tuning layer ─────────────────────────────────────
        if opts.include_tuning {
            if let Ok(Some(prompt)) = prompt_tuning::get_tuned_prompt(&*store, nar_id) {
                layers.tuning = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}tuning/tuned_prompt.json"),
                    &prompt,
                    opts.pretty,
                )?;
            }
        }

        // ── Embeddings layer ─────────────────────────────────
        if opts.include_embeddings {
            let mut index = EmbeddingIndex {
                dimension: 0,
                entities: std::collections::HashMap::new(),
                situations: std::collections::HashMap::new(),
                chunks: std::collections::HashMap::new(),
            };

            // Entity embeddings
            if !entity_embeddings_saved.is_empty() {
                if index.dimension == 0 {
                    index.dimension = entity_embeddings_saved[0].1.len();
                }
                let bin_name = "entity_embeddings.bin";
                let bin_path = format!("{prefix}embeddings/{bin_name}");
                let mut bin_data = Vec::new();
                for (id, emb) in &entity_embeddings_saved {
                    let offset = bin_data.len();
                    for &f in emb {
                        bin_data.extend_from_slice(&f.to_le_bytes());
                    }
                    index.entities.insert(
                        id.to_string(),
                        EmbeddingEntry {
                            file: bin_name.to_string(),
                            offset,
                        },
                    );
                }
                write_bin(&mut zip, &file_opts, &bin_path, &bin_data)?;
            }

            // Situation embeddings
            if !sit_embeddings_saved.is_empty() {
                if index.dimension == 0 {
                    index.dimension = sit_embeddings_saved[0].1.len();
                }
                let bin_name = "situation_embeddings.bin";
                let bin_path = format!("{prefix}embeddings/{bin_name}");
                let mut bin_data = Vec::new();
                for (id, emb) in &sit_embeddings_saved {
                    let offset = bin_data.len();
                    for &f in emb {
                        bin_data.extend_from_slice(&f.to_le_bytes());
                    }
                    index.situations.insert(
                        id.to_string(),
                        EmbeddingEntry {
                            file: bin_name.to_string(),
                            offset,
                        },
                    );
                }
                write_bin(&mut zip, &file_opts, &bin_path, &bin_data)?;
            }

            if !index.entities.is_empty() || !index.situations.is_empty() {
                layers.embeddings = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}embeddings/embedding_index.json"),
                    &index,
                    opts.pretty,
                )?;
            }
        }

        // ── Annotations layer (v1.1.0) ───────────────────────
        // Skill output from `/tensa-narrative-llm <nid> dramatic-irony` etc.
        // lands here. Uses the bucketed scan helper rather than per-situation
        // calls — one O(|ann/|) prefix scan instead of N round-trips.
        if opts.include_annotations {
            if let Ok(buckets) =
                crate::writer::annotation::list_annotations_for_scenes(&*store, &situation_ids)
            {
                let mut annotations: Vec<_> =
                    buckets.into_values().flat_map(|v| v.into_iter()).collect();
                annotations.sort_by_key(|a| (a.situation_id, a.span.0));
                if !annotations.is_empty() {
                    layers.annotations = true;
                    write_json(
                        &mut zip,
                        &file_opts,
                        &format!("{prefix}annotations/annotations.json"),
                        &annotations,
                        opts.pretty,
                    )?;
                }
            }
        }

        // ── Pinned facts layer (v1.1.0) ──────────────────────
        // What `/tensa-narrative-llm <nid> commitments` writes.
        if opts.include_pinned_facts {
            if let Ok(facts) = crate::narrative::continuity::list_pinned_facts(&*store, nar_id) {
                if !facts.is_empty() {
                    layers.pinned_facts = true;
                    write_json(
                        &mut zip,
                        &file_opts,
                        &format!("{prefix}pinned_facts/pinned_facts.json"),
                        &facts,
                        opts.pretty,
                    )?;
                }
            }
        }

        // ── Revisions layer (v1.1.0) ─────────────────────────
        // What `commit_narrative_revision` (called by `narrative-diagnose-and-repair`)
        // writes. We export the FULL revisions (snapshot + author + message),
        // not just the per-narrative summary index, so a re-imported archive
        // can restore any historical state.
        if opts.include_revisions {
            if let Ok(summaries) = crate::narrative::revision::list_revisions(&*store, nar_id) {
                let mut revisions = Vec::with_capacity(summaries.len());
                for summary in &summaries {
                    if let Ok(rev) = crate::narrative::revision::get_revision(&*store, &summary.id)
                    {
                        revisions.push(rev);
                    }
                }
                if !revisions.is_empty() {
                    layers.revisions = true;
                    write_json(
                        &mut zip,
                        &file_opts,
                        &format!("{prefix}revisions/revisions.json"),
                        &revisions,
                        opts.pretty,
                    )?;
                }
            }
        }

        // ── Workshop reports layer (v1.1.0) ──────────────────
        if opts.include_workshop_reports {
            if let Ok(summaries) = crate::narrative::workshop::list_reports(&*store, nar_id) {
                let mut reports = Vec::with_capacity(summaries.len());
                for summary in &summaries {
                    if let Ok(report) =
                        crate::narrative::workshop::get_report(&*store, &summary.id)
                    {
                        reports.push(report);
                    }
                }
                if !reports.is_empty() {
                    layers.workshop_reports = true;
                    write_json(
                        &mut zip,
                        &file_opts,
                        &format!("{prefix}workshop_reports/workshop_reports.json"),
                        &reports,
                        opts.pretty,
                    )?;
                }
            }
        }

        // ── Narrative plan layer (v1.1.0) ────────────────────
        if opts.include_narrative_plan {
            if let Ok(Some(plan)) = crate::narrative::plan::get_plan(&*store, nar_id) {
                layers.narrative_plan = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("{prefix}plan/narrative_plan.json"),
                    &plan,
                    opts.pretty,
                )?;
            }
        }

        // ── Analysis-status layer (v1.1.0) ───────────────────
        // Preserves the lock state — without this, a re-imported archive
        // would lose `Source: Skill` + `locked: true` rows and a subsequent
        // bulk-analysis run would silently overwrite skill-attested results.
        if opts.include_analysis_status {
            let status_store =
                crate::analysis_status::AnalysisStatusStore::new(store.clone());
            if let Ok(rows) = status_store.list_for_narrative(nar_id) {
                if !rows.is_empty() {
                    layers.analysis_status = true;
                    write_json(
                        &mut zip,
                        &file_opts,
                        &format!("{prefix}analysis_status/entries.json"),
                        &rows,
                        opts.pretty,
                    )?;
                }
            }
        }
    }

    // ── Taxonomy layer ───────────────────────────────────────
    if opts.include_taxonomy {
        let tax_registry = TaxonomyRegistry::new(store.clone());
        let mut custom_entries: Vec<TaxonomyEntry> = Vec::new();
        // Export custom entries for known categories
        for category in &["genre", "content_type"] {
            if let Ok(entries) = tax_registry.list(category) {
                for entry in entries {
                    if !entry.is_builtin {
                        custom_entries.push(entry);
                    }
                }
            }
        }
        if !custom_entries.is_empty() {
            layers.taxonomy = true;
            write_json(
                &mut zip,
                &file_opts,
                "taxonomy/taxonomy.json",
                &custom_entries,
                opts.pretty,
            )?;
        }
    }

    // ── Projects layer ───────────────────────────────────────
    if opts.include_projects && !project_ids.is_empty() {
        let proj_registry = ProjectRegistry::new(store.clone());
        for pid in &project_ids {
            if let Ok(project) = proj_registry.get(pid) {
                layers.projects = true;
                write_json(
                    &mut zip,
                    &file_opts,
                    &format!("projects/{pid}.json"),
                    &project,
                    opts.pretty,
                )?;
            }
        }
    }

    // ── Manifest ─────────────────────────────────────────────
    let manifest = ArchiveManifest {
        tensa_archive_version: ARCHIVE_VERSION.to_string(),
        created_at: Utc::now(),
        created_by: Some(ArchiveCreatedBy {
            tool: "tensa".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        narratives: narrative_ids.iter().map(|s| s.to_string()).collect(),
        layers,
        strict_references: true,
        id_namespace: None,
        description: None,
    };
    write_json(
        &mut zip,
        &file_opts,
        "manifest.json",
        &manifest,
        opts.pretty,
    )?;

    let cursor = zip
        .finish()
        .map_err(|e| TensaError::ExportError(format!("ZIP finalize: {e}")))?;
    Ok(cursor.into_inner())
}

/// Scan all inference results from the KV store.
fn scan_inference_results(
    store: &dyn crate::store::KVStore,
) -> Result<Vec<crate::types::InferenceResult>> {
    let prefix = b"ir/".to_vec();
    let pairs = store.prefix_scan(&prefix)?;
    let mut results = Vec::new();
    for (_key, value) in pairs {
        if let Ok(r) = serde_json::from_slice(&value) {
            results.push(r);
        }
    }
    Ok(results)
}

/// Helper: write raw binary data to the ZIP.
fn write_bin(
    zip: &mut ZipWriter<Cursor<Vec<u8>>>,
    opts: &SimpleFileOptions,
    path: &str,
    data: &[u8],
) -> Result<()> {
    zip.start_file(path, *opts)
        .map_err(|e| TensaError::ExportError(format!("ZIP write {path}: {e}")))?;
    zip.write_all(data)
        .map_err(|e| TensaError::ExportError(format!("ZIP write {path}: {e}")))?;
    Ok(())
}

/// Helper: serialize a value as JSON and write it to the ZIP.
fn write_json<T: serde::Serialize>(
    zip: &mut ZipWriter<Cursor<Vec<u8>>>,
    opts: &SimpleFileOptions,
    path: &str,
    value: &T,
    pretty: bool,
) -> Result<()> {
    let bytes = if pretty {
        serde_json::to_vec_pretty(value)
    } else {
        serde_json::to_vec(value)
    }
    .map_err(|e| TensaError::ExportError(format!("JSON serialization for {path}: {e}")))?;

    zip.start_file(path, *opts)
        .map_err(|e| TensaError::ExportError(format!("ZIP write {path}: {e}")))?;
    zip.write_all(&bytes)
        .map_err(|e| TensaError::ExportError(format!("ZIP write {path}: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::Hypergraph;
    use crate::narrative::registry::NarrativeRegistry;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use crate::types::*;
    use chrono::Utc;
    use std::io::Read;
    use std::sync::Arc;
    use uuid::Uuid;

    fn setup_test_narrative(store: Arc<MemoryStore>) -> Hypergraph {
        let hg = Hypergraph::new(store.clone());
        let nar_reg = NarrativeRegistry::new(store.clone() as Arc<dyn crate::store::KVStore>);

        let now = Utc::now();
        let narrative = Narrative {
            id: "test-nar".to_string(),
            title: "Test Narrative".to_string(),
            genre: Some("novel".to_string()),
            tags: vec!["test".to_string()],
            source: None,
            project_id: None,
            description: None,
            authors: vec![],
            language: None,
            publication_date: None,
            cover_url: None,
            custom_properties: std::collections::HashMap::new(),
            entity_count: 0,
            situation_count: 0,
            created_at: now,
            updated_at: now,
        };
        nar_reg.create(narrative).unwrap();

        let entity = Entity {
            id: Uuid::now_v7(),
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Alice"}),
            beliefs: None,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            provenance: vec![],
            extraction_method: Some(ExtractionMethod::LlmParsed),
            narrative_id: Some("test-nar".to_string()),
            created_at: now,
            updated_at: now,
            deleted_at: None,
            transaction_time: None,
        };
        hg.create_entity(entity).unwrap();

        let situation = Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some("Test Event".to_string()),
            description: Some("Something happened".to_string()),
            temporal: AllenInterval {
                start: Some(now),
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
            raw_content: vec![ContentBlock::text("Something happened in the story.")],
            narrative_level: NarrativeLevel::Scene,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.8,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::LlmParsed,
            provenance: vec![],
            narrative_id: Some("test-nar".to_string()),
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
        hg.create_situation(situation).unwrap();

        hg
    }

    /// Read a file from the ZIP archive as a string.
    fn read_zip_str(archive: &mut zip::ZipArchive<Cursor<Vec<u8>>>, name: &str) -> String {
        let mut file = archive.by_name(name).unwrap();
        let mut s = String::new();
        file.read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn test_export_archive_basic() {
        let store = Arc::new(MemoryStore::new());
        let hg = setup_test_narrative(store.clone());

        let opts = ArchiveExportOptions {
            include_embeddings: false,
            ..Default::default()
        };
        let bytes = export_archive(&["test-nar"], &hg, &opts).unwrap();
        assert!(!bytes.is_empty());

        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor).unwrap();

        // Check manifest
        let manifest: ArchiveManifest =
            serde_json::from_str(&read_zip_str(&mut archive, "manifest.json")).unwrap();
        assert_eq!(manifest.tensa_archive_version, ARCHIVE_VERSION);
        assert_eq!(manifest.narratives, vec!["test-nar"]);
        assert!(manifest.layers.core);

        // Check narrative.json
        let nar: crate::narrative::types::Narrative = serde_json::from_str(&read_zip_str(
            &mut archive,
            "narratives/test-nar/narrative.json",
        ))
        .unwrap();
        assert_eq!(nar.title, "Test Narrative");

        // Check entities.json
        let entities: Vec<Entity> = serde_json::from_str(&read_zip_str(
            &mut archive,
            "narratives/test-nar/entities.json",
        ))
        .unwrap();
        assert_eq!(entities.len(), 1);
        assert!(entities[0].embedding.is_none()); // Stripped

        // Check situations.json
        let situations: Vec<Situation> = serde_json::from_str(&read_zip_str(
            &mut archive,
            "narratives/test-nar/situations.json",
        ))
        .unwrap();
        assert_eq!(situations.len(), 1);

        // Check participations.json exists
        assert!(archive
            .by_name("narratives/test-nar/participations.json")
            .is_ok());
    }

    #[test]
    fn test_export_archive_with_embeddings() {
        let store = Arc::new(MemoryStore::new());
        let hg = setup_test_narrative(store.clone());

        let opts = ArchiveExportOptions {
            include_embeddings: true,
            include_sources: false,
            include_chunks: false,
            include_state_versions: false,
            include_inference: false,
            include_analysis: false,
            include_tuning: false,
            include_taxonomy: false,
            include_projects: false,
            include_annotations: false,
            include_pinned_facts: false,
            include_revisions: false,
            include_workshop_reports: false,
            include_narrative_plan: false,
            include_analysis_status: false,
            pretty: false,
            include_synthetic: false,
        };
        let bytes = export_archive(&["test-nar"], &hg, &opts).unwrap();

        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor).unwrap();

        // Entities should retain embeddings in JSON
        let entities: Vec<Entity> = serde_json::from_str(&read_zip_str(
            &mut archive,
            "narratives/test-nar/entities.json",
        ))
        .unwrap();
        assert!(entities[0].embedding.is_some());

        // Binary embedding file should exist
        assert!(archive
            .by_name("narratives/test-nar/embeddings/entity_embeddings.bin")
            .is_ok());
        assert!(archive
            .by_name("narratives/test-nar/embeddings/embedding_index.json")
            .is_ok());
    }

    #[test]
    fn test_export_archive_empty_narrative() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let nar_reg = NarrativeRegistry::new(store.clone() as Arc<dyn crate::store::KVStore>);

        let now = Utc::now();
        nar_reg
            .create(Narrative {
                id: "empty".to_string(),
                title: "Empty".to_string(),
                genre: None,
                tags: vec![],
                source: None,
                project_id: None,
                description: None,
                authors: vec![],
                language: None,
                publication_date: None,
                cover_url: None,
                custom_properties: std::collections::HashMap::new(),
                entity_count: 0,
                situation_count: 0,
                created_at: now,
                updated_at: now,
            })
            .unwrap();

        let opts = ArchiveExportOptions::default();
        let bytes = export_archive(&["empty"], &hg, &opts).unwrap();

        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor).unwrap();

        // Should have manifest + narrative.json + entities + situations + participations
        let mut file_names = Vec::new();
        for i in 0..archive.len() {
            let f = archive.by_index(i).unwrap();
            file_names.push(f.name().to_string());
        }
        assert!(file_names.contains(&"manifest.json".to_string()));
        assert!(file_names.contains(&"narratives/empty/narrative.json".to_string()));
        assert!(file_names.contains(&"narratives/empty/entities.json".to_string()));
    }

    #[test]
    fn test_export_archive_multiple_narratives() {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let nar_reg = NarrativeRegistry::new(store.clone() as Arc<dyn crate::store::KVStore>);

        let now = Utc::now();
        for slug in &["nar-a", "nar-b"] {
            nar_reg
                .create(Narrative {
                    id: slug.to_string(),
                    title: slug.to_string(),
                    genre: None,
                    tags: vec![],
                    source: None,
                    project_id: None,
                    description: None,
                    authors: vec![],
                    language: None,
                    publication_date: None,
                    cover_url: None,
                    custom_properties: std::collections::HashMap::new(),
                    entity_count: 0,
                    situation_count: 0,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let opts = ArchiveExportOptions::default();
        let bytes = export_archive(&["nar-a", "nar-b"], &hg, &opts).unwrap();

        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor).unwrap();
        let file_count = archive.len();
        let mut file_names = Vec::new();
        for i in 0..file_count {
            let f = archive.by_index(i).unwrap();
            file_names.push(f.name().to_string());
        }

        assert!(file_names.contains(&"narratives/nar-a/narrative.json".to_string()));
        assert!(file_names.contains(&"narratives/nar-b/narrative.json".to_string()));

        // Manifest should list both
        let manifest: ArchiveManifest =
            serde_json::from_str(&read_zip_str(&mut archive, "manifest.json")).unwrap();
        assert_eq!(manifest.narratives.len(), 2);
    }
}
