//! Tensa Narrative Archive (.tensa) — import logic.
//!
//! Reads a ZIP archive and imports narratives, entities, situations,
//! participations, and optional layers into the hypergraph.

use std::collections::HashMap;
use std::io::{Cursor, Read};

use chrono::Utc;
use uuid::Uuid;

use crate::analysis::community;
use crate::error::{Result, TensaError};
use crate::export::archive_types::*;
use crate::hypergraph::Hypergraph;
use crate::inference::jobs::JobQueue;
use crate::ingestion::prompt_tuning;
use crate::narrative::project::ProjectRegistry;
use crate::narrative::registry::NarrativeRegistry;
use crate::narrative::taxonomy::TaxonomyRegistry;
use crate::narrative::types::{Narrative, TaxonomyEntry};
use crate::source::{ContentionLink, Source, SourceAttribution};
use crate::types::*;

/// Import a `.tensa` archive into the hypergraph.
pub fn import_archive(
    data: &[u8],
    hypergraph: &Hypergraph,
    opts: &ArchiveImportOptions,
) -> Result<ArchiveImportReport> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| TensaError::Internal(format!("Invalid ZIP archive: {e}")))?;

    // ── Read manifest ────────────────────────────────────────
    let manifest: ArchiveManifest = read_json_file(&mut archive, "manifest.json")?;

    // Version check: only support 1.x.x
    let version_parts: Vec<&str> = manifest.tensa_archive_version.split('.').collect();
    if version_parts.first() != Some(&"1") {
        return Err(TensaError::Internal(format!(
            "Unsupported archive version: {}. This reader supports 1.x.x.",
            manifest.tensa_archive_version
        )));
    }

    let strict = opts.strict_references && manifest.strict_references;
    let store = hypergraph.store_arc();
    let mut report = ArchiveImportReport::default();
    let mut remap: HashMap<Uuid, Uuid> = HashMap::new();

    // ── Taxonomy ─────────────────────────────────────────────
    if manifest.layers.taxonomy {
        if let Ok(entries) =
            read_json_file::<Vec<TaxonomyEntry>>(&mut archive, "taxonomy/taxonomy.json")
        {
            let tax_registry = TaxonomyRegistry::new(store.clone());
            for entry in entries {
                if !entry.is_builtin {
                    if tax_registry.add(entry).is_ok() {
                        report.taxonomy_entries_created += 1;
                    }
                }
            }
        }
    }

    // ── Per-narrative import ─────────────────────────────────
    for slug in &manifest.narratives {
        let target_slug = opts.target_narrative_id.as_deref().unwrap_or(slug.as_str());
        let prefix = format!("narratives/{slug}/");

        // Narrative metadata — lenient parsing for externally-created archives
        let nar_registry = NarrativeRegistry::new(store.clone());
        let mut narrative: Narrative =
            read_narrative_lenient(&mut archive, &format!("{prefix}narrative.json"), slug)?;
        narrative.id = target_slug.to_string();

        // Import referenced project (before narrative, so project_id reference is valid)
        if manifest.layers.projects {
            if let Some(ref pid) = narrative.project_id {
                let path = format!("projects/{pid}.json");
                if let Ok(project) =
                    read_json_file::<crate::narrative::types::Project>(&mut archive, &path)
                {
                    let proj_registry = ProjectRegistry::new(store.clone());
                    match proj_registry.create(project) {
                        Ok(_) => report.projects_created += 1,
                        Err(_) => {} // Already exists, skip
                    }
                }
            }
        }

        match nar_registry.create(narrative.clone()) {
            Ok(_) => report.narratives_imported += 1,
            Err(TensaError::NarrativeExists(_)) => {
                // Narrative already exists — update or skip based on merge_mode
                if opts.merge_mode {
                    report.warnings.push(format!(
                        "Narrative '{target_slug}' already exists, merging into it"
                    ));
                } else {
                    // Try with suffix
                    let mut n = 2;
                    loop {
                        let suffixed = format!("{target_slug}-{n}");
                        narrative.id = suffixed.clone();
                        match nar_registry.create(narrative.clone()) {
                            Ok(_) => {
                                report.narratives_imported += 1;
                                report.warnings.push(format!(
                                    "Narrative '{target_slug}' already existed, imported as '{suffixed}'"
                                ));
                                break;
                            }
                            Err(TensaError::NarrativeExists(_)) => {
                                n += 1;
                            }
                            Err(e) => return Err(e),
                        }
                        if n > 100 {
                            return Err(TensaError::Internal(format!(
                                "Could not find available slug for narrative '{target_slug}'"
                            )));
                        }
                    }
                }
            }
            Err(e) => return Err(e),
        }
        let final_slug = narrative.id.clone();

        // ── Sources ──────────────────────────────────────────
        if manifest.layers.sources {
            if let Ok(sources) = read_json_file::<Vec<Source>>(
                &mut archive,
                &format!("{prefix}sources/sources.json"),
            ) {
                for source in sources {
                    let old_id = source.id;
                    match hypergraph.create_source(source.clone()) {
                        Ok(_) => report.sources_created += 1,
                        Err(TensaError::SourceExists(_)) => {
                            if !opts.merge_mode {
                                let new_id = Uuid::now_v7();
                                let mut remapped = source;
                                remapped.id = new_id;
                                remap.insert(old_id, new_id);
                                if hypergraph.create_source(remapped).is_ok() {
                                    report.sources_created += 1;
                                }
                            }
                        }
                        Err(e) => report.errors.push(format!("Source {old_id}: {e}")),
                    }
                }
            }
        }

        // ── Entities — lenient parsing fills missing defaults ─
        let entities: Vec<Entity> =
            read_entities_lenient(&mut archive, &format!("{prefix}entities.json"))
                .unwrap_or_default();

        for mut entity in entities {
            let old_id = entity.id;
            entity.narrative_id = Some(final_slug.clone());

            // Check for UUID clash
            if hypergraph.get_entity(&entity.id).is_ok() {
                if opts.merge_mode {
                    report.entities_skipped += 1;
                    continue;
                }
                let new_id = Uuid::now_v7();
                remap.insert(old_id, new_id);
                entity.id = new_id;
            }

            match hypergraph.create_entity(entity) {
                Ok(_) => report.entities_created += 1,
                Err(e) => report.errors.push(format!("Entity {old_id}: {e}")),
            }
        }

        // ── Situations — lenient parsing fills missing defaults
        let situations: Vec<Situation> =
            read_situations_lenient(&mut archive, &format!("{prefix}situations.json"))
                .unwrap_or_default();

        for mut situation in situations {
            let old_id = situation.id;
            situation.narrative_id = Some(final_slug.clone());

            // Remap Allen relation targets
            for rel in &mut situation.temporal.relations {
                if let Some(&new_id) = remap.get(&rel.target_situation) {
                    rel.target_situation = new_id;
                }
            }
            // Remap spatial location_entity
            if let Some(ref mut spatial) = situation.spatial {
                if let Some(ref mut loc_ent) = spatial.location_entity {
                    if let Some(&new_id) = remap.get(loc_ent) {
                        *loc_ent = new_id;
                    }
                }
            }
            // Remap discourse focalization
            if let Some(ref mut disc) = situation.discourse {
                if let Some(ref mut foc) = disc.focalization {
                    if let Some(&new_id) = remap.get(foc) {
                        *foc = new_id;
                    }
                }
            }
            // Remap causal link references in embedded causes
            for cause in &mut situation.causes {
                if let Some(&new_id) = remap.get(&cause.from_situation) {
                    cause.from_situation = new_id;
                }
                if let Some(&new_id) = remap.get(&cause.to_situation) {
                    cause.to_situation = new_id;
                }
            }

            // Check for UUID clash
            if hypergraph.get_situation(&situation.id).is_ok() {
                if opts.merge_mode {
                    report.situations_skipped += 1;
                    continue;
                }
                let new_id = Uuid::now_v7();
                remap.insert(old_id, new_id);
                situation.id = new_id;
            }

            match hypergraph.create_situation(situation) {
                Ok(_) => report.situations_created += 1,
                Err(e) => report.errors.push(format!("Situation {old_id}: {e}")),
            }
        }

        // ── Participations ───────────────────────────────────
        let participations: Vec<Participation> =
            match read_json_file(&mut archive, &format!("{prefix}participations.json")) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("Failed to parse participations.json: {e}");
                    report
                        .warnings
                        .push(format!("Failed to parse participations.json: {e}"));
                    vec![]
                }
            };

        tracing::info!(
            "Archive contains {} participations to import",
            participations.len()
        );
        for mut part in participations {
            // Remap entity/situation IDs
            if let Some(&new_id) = remap.get(&part.entity_id) {
                part.entity_id = new_id;
            }
            if let Some(&new_id) = remap.get(&part.situation_id) {
                part.situation_id = new_id;
            }
            // Remap knowledge facts
            if let Some(ref mut info) = part.info_set {
                remap_knowledge_facts(&remap, &mut info.knows_before);
                remap_knowledge_facts(&remap, &mut info.learns);
                remap_knowledge_facts(&remap, &mut info.reveals);
                for belief in &mut info.beliefs_about_others {
                    if let Some(&new_id) = remap.get(&belief.about_entity) {
                        belief.about_entity = new_id;
                    }
                    if let Some(&new_id) = remap.get(&belief.last_updated_at) {
                        belief.last_updated_at = new_id;
                    }
                    remap_knowledge_facts(&remap, &mut belief.believed_knowledge);
                }
            }

            // Validate references
            if hypergraph.get_entity(&part.entity_id).is_err() {
                if strict {
                    report.errors.push(format!(
                        "Participation references missing entity {}",
                        part.entity_id
                    ));
                } else {
                    report.warnings.push(format!(
                        "Participation references missing entity {}",
                        part.entity_id
                    ));
                }
                continue;
            }
            if hypergraph.get_situation(&part.situation_id).is_err() {
                if strict {
                    report.errors.push(format!(
                        "Participation references missing situation {}",
                        part.situation_id
                    ));
                } else {
                    report.warnings.push(format!(
                        "Participation references missing situation {}",
                        part.situation_id
                    ));
                }
                continue;
            }

            match hypergraph.add_participant(part) {
                Ok(_) => report.participations_created += 1,
                Err(e) => report.errors.push(format!("Participation: {e}")),
            }
        }

        // ── Causal links ─────────────────────────────────────
        if let Ok(links) =
            read_json_file::<Vec<CausalLink>>(&mut archive, &format!("{prefix}causal_links.json"))
        {
            for mut link in links {
                if let Some(&new_id) = remap.get(&link.from_situation) {
                    link.from_situation = new_id;
                }
                if let Some(&new_id) = remap.get(&link.to_situation) {
                    link.to_situation = new_id;
                }
                match hypergraph.add_causal_link(link) {
                    Ok(_) => report.causal_links_created += 1,
                    Err(e) => report.warnings.push(format!("Causal link: {e}")),
                }
            }
        }

        // ── Source attributions ───────────────────────────────
        if manifest.layers.sources {
            if let Ok(attributions) = read_json_file::<Vec<SourceAttribution>>(
                &mut archive,
                &format!("{prefix}sources/attributions.json"),
            ) {
                for mut attr in attributions {
                    if let Some(&new_id) = remap.get(&attr.source_id) {
                        attr.source_id = new_id;
                    }
                    if let Some(&new_id) = remap.get(&attr.target_id) {
                        attr.target_id = new_id;
                    }
                    match hypergraph.add_attribution(attr) {
                        Ok(_) => report.attributions_created += 1,
                        Err(e) => report.warnings.push(format!("Attribution: {e}")),
                    }
                }
            }

            if let Ok(contentions) = read_json_file::<Vec<ContentionLink>>(
                &mut archive,
                &format!("{prefix}sources/contentions.json"),
            ) {
                for mut link in contentions {
                    if let Some(&new_id) = remap.get(&link.situation_a) {
                        link.situation_a = new_id;
                    }
                    if let Some(&new_id) = remap.get(&link.situation_b) {
                        link.situation_b = new_id;
                    }
                    match hypergraph.add_contention(link) {
                        Ok(_) => report.contentions_created += 1,
                        Err(e) => report.warnings.push(format!("Contention: {e}")),
                    }
                }
            }
        }

        // ── Chunks ───────────────────────────────────────────
        if manifest.layers.chunks {
            if let Ok(chunks) = read_json_file::<Vec<ChunkRecord>>(
                &mut archive,
                &format!("{prefix}chunks/chunks.json"),
            ) {
                for mut chunk in chunks {
                    let old_id = chunk.id;
                    if let Some(&new_id) = remap.get(&old_id) {
                        chunk.id = new_id;
                    }
                    chunk.narrative_id = Some(final_slug.clone());
                    match hypergraph.store_chunk(&chunk) {
                        Ok(_) => report.chunks_created += 1,
                        Err(e) => report.warnings.push(format!("Chunk {old_id}: {e}")),
                    }
                }
            }
        }

        // ── State versions ───────────────────────────────────
        if manifest.layers.state_versions {
            if let Ok(versions) = read_json_file::<Vec<StateVersion>>(
                &mut archive,
                &format!("{prefix}state_versions/state_versions.json"),
            ) {
                for mut sv in versions {
                    if let Some(&new_id) = remap.get(&sv.entity_id) {
                        sv.entity_id = new_id;
                    }
                    if let Some(&new_id) = remap.get(&sv.situation_id) {
                        sv.situation_id = new_id;
                    }
                    match hypergraph.create_state_version(sv) {
                        Ok(_) => report.state_versions_created += 1,
                        Err(e) => report.warnings.push(format!("StateVersion: {e}")),
                    }
                }
            }
        }

        // ── Inference results ────────────────────────────────
        if manifest.layers.inference {
            if let Ok(results) = read_json_file::<Vec<InferenceResult>>(
                &mut archive,
                &format!("{prefix}inference/results.json"),
            ) {
                let job_queue = JobQueue::new(store.clone());
                for mut result in results {
                    if let Some(&new_id) = remap.get(&result.target_id) {
                        result.target_id = new_id;
                    }
                    match job_queue.store_result(result) {
                        Ok(_) => report.inference_results_created += 1,
                        Err(e) => report.warnings.push(format!("InferenceResult: {e}")),
                    }
                }
            }
        }

        // ── Analysis (communities) ───────────────────────────
        if manifest.layers.analysis {
            if let Ok(summaries) = read_json_file::<Vec<crate::analysis::community::CommunitySummary>>(
                &mut archive,
                &format!("{prefix}analysis/communities.json"),
            ) {
                for mut cs in summaries {
                    cs.narrative_id = final_slug.clone();
                    // Remap entity IDs
                    cs.entity_ids = cs
                        .entity_ids
                        .iter()
                        .map(|id| remap.get(id).copied().unwrap_or(*id))
                        .collect();
                    match community::store_summary(&*store, &cs) {
                        Ok(_) => report.communities_created += 1,
                        Err(e) => report.warnings.push(format!("CommunitySummary: {e}")),
                    }
                }
            }
        }

        // ── Tuned prompts ────────────────────────────────────
        if manifest.layers.tuning {
            if let Ok(prompt) = read_json_file::<crate::ingestion::prompt_tuning::TunedPrompt>(
                &mut archive,
                &format!("{prefix}tuning/tuned_prompt.json"),
            ) {
                let mut p = prompt;
                p.narrative_id = final_slug.clone();
                match prompt_tuning::store_tuned_prompt(&*store, &p) {
                    Ok(_) => report.prompts_created += 1,
                    Err(e) => report.warnings.push(format!("TunedPrompt: {e}")),
                }
            }
        }

        // ── Annotations (v1.1.0) ─────────────────────────────
        if manifest.layers.annotations {
            if let Ok(annotations) = read_json_file::<Vec<crate::writer::annotation::Annotation>>(
                &mut archive,
                &format!("{prefix}annotations/annotations.json"),
            ) {
                for mut ann in annotations {
                    if let Some(&new_id) = remap.get(&ann.situation_id) {
                        ann.situation_id = new_id;
                    }
                    // create_annotation re-keys against ann.id; ID clashes are
                    // not a concern across narratives so we keep the originals.
                    match crate::writer::annotation::create_annotation(&*store, ann) {
                        Ok(_) => report.annotations_created += 1,
                        Err(e) => report.warnings.push(format!("Annotation: {e}")),
                    }
                }
            }
        }

        // ── Pinned facts (v1.1.0) ────────────────────────────
        if manifest.layers.pinned_facts {
            if let Ok(facts) = read_json_file::<Vec<crate::types::PinnedFact>>(
                &mut archive,
                &format!("{prefix}pinned_facts/pinned_facts.json"),
            ) {
                for mut fact in facts {
                    fact.narrative_id = final_slug.clone();
                    if let Some(ref mut entity_id) = fact.entity_id {
                        if let Some(&new_id) = remap.get(entity_id) {
                            *entity_id = new_id;
                        }
                    }
                    match crate::narrative::continuity::create_pinned_fact(&*store, fact) {
                        Ok(_) => report.pinned_facts_created += 1,
                        Err(e) => report.warnings.push(format!("PinnedFact: {e}")),
                    }
                }
            }
        }

        // ── Revisions (v1.1.0) ───────────────────────────────
        // Stored verbatim so revision history (snapshot + author + message +
        // parent chain) survives the round-trip. The snapshot inside each
        // revision references entity / situation IDs that may have been
        // remapped during core import; we cannot rewrite serialized snapshot
        // bytes safely, so when remap is non-empty we emit a warning and
        // keep the original snapshot intact (revisions are advisory history,
        // not load-bearing for the live graph).
        if manifest.layers.revisions {
            if let Ok(revisions) = read_json_file::<Vec<crate::types::NarrativeRevision>>(
                &mut archive,
                &format!("{prefix}revisions/revisions.json"),
            ) {
                let warned_remap = !remap.is_empty();
                for mut rev in revisions {
                    rev.narrative_id = final_slug.clone();
                    let key = crate::hypergraph::keys::revision_key(&rev.id);
                    let idx_key = crate::hypergraph::keys::revision_narrative_index_key(
                        &rev.narrative_id,
                        &rev.id,
                    );
                    let bytes = match serde_json::to_vec(&rev) {
                        Ok(b) => b,
                        Err(e) => {
                            report.warnings.push(format!("Revision serialize: {e}"));
                            continue;
                        }
                    };
                    let res = store
                        .put(&key, &bytes)
                        .and_then(|_| store.put(&idx_key, rev.id.as_bytes()));
                    match res {
                        Ok(_) => report.revisions_created += 1,
                        Err(e) => report.warnings.push(format!("Revision: {e}")),
                    }
                }
                if warned_remap && report.revisions_created > 0 {
                    report.warnings.push(
                        "Revisions imported with original UUIDs in their snapshots — \
                         restoring an old revision may reference remapped entities/situations."
                            .into(),
                    );
                }
            }
        }

        // ── Workshop reports (v1.1.0) ────────────────────────
        if manifest.layers.workshop_reports {
            if let Ok(reports) = read_json_file::<Vec<crate::narrative::workshop::WorkshopReport>>(
                &mut archive,
                &format!("{prefix}workshop_reports/workshop_reports.json"),
            ) {
                for mut wr in reports {
                    wr.narrative_id = final_slug.clone();
                    let key = crate::hypergraph::keys::workshop_report_key(&wr.id);
                    let idx_key = crate::hypergraph::keys::workshop_report_narrative_index_key(
                        &wr.narrative_id,
                        &wr.id,
                    );
                    let bytes = match serde_json::to_vec(&wr) {
                        Ok(b) => b,
                        Err(e) => {
                            report.warnings.push(format!("WorkshopReport serialize: {e}"));
                            continue;
                        }
                    };
                    let res = store
                        .put(&key, &bytes)
                        .and_then(|_| store.put(&idx_key, wr.id.as_bytes()));
                    match res {
                        Ok(_) => report.workshop_reports_created += 1,
                        Err(e) => report.warnings.push(format!("WorkshopReport: {e}")),
                    }
                }
            }
        }

        // ── Narrative plan (v1.1.0) ──────────────────────────
        if manifest.layers.narrative_plan {
            if let Ok(plan) = read_json_file::<crate::types::NarrativePlan>(
                &mut archive,
                &format!("{prefix}plan/narrative_plan.json"),
            ) {
                let mut p = plan;
                p.narrative_id = final_slug.clone();
                match crate::narrative::plan::upsert_plan(&*store, p) {
                    Ok(_) => report.narrative_plans_created += 1,
                    Err(e) => report.warnings.push(format!("NarrativePlan: {e}")),
                }
            }
        }

        // ── Analysis-status registry (v1.1.0) ────────────────
        if manifest.layers.analysis_status {
            if let Ok(entries) = read_json_file::<
                Vec<crate::analysis_status::AnalysisStatusEntry>,
            >(
                &mut archive,
                &format!("{prefix}analysis_status/entries.json"),
            ) {
                let status_store =
                    crate::analysis_status::AnalysisStatusStore::new(store.clone());
                for mut entry in entries {
                    entry.narrative_id = final_slug.clone();
                    match status_store.upsert(&entry) {
                        Ok(_) => report.analysis_status_entries_created += 1,
                        Err(e) => report.warnings.push(format!("AnalysisStatusEntry: {e}")),
                    }
                }
            }
        }
    }

    // Build remap string map for report
    for (old, new) in &remap {
        report.id_remaps.insert(old.to_string(), new.to_string());
    }

    Ok(report)
}

/// Read a raw JSON Value from the ZIP archive.
fn read_json_value(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    path: &str,
) -> Result<serde_json::Value> {
    let mut file = archive
        .by_name(path)
        .map_err(|e| TensaError::Internal(format!("Missing archive file '{path}': {e}")))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .map_err(|e| TensaError::Internal(format!("Failed to read '{path}': {e}")))?;
    serde_json::from_str(&buf)
        .map_err(|e| TensaError::Internal(format!("Failed to parse '{path}': {e}")))
}

/// Read narrative.json leniently — fills missing required fields with defaults.
/// External tools often omit `id`, `entity_count`, `created_at`, etc.
fn read_narrative_lenient(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    path: &str,
    slug: &str,
) -> Result<Narrative> {
    let mut v = read_json_value(archive, path)?;
    let obj = v
        .as_object_mut()
        .ok_or_else(|| TensaError::Internal(format!("{path}: expected JSON object")))?;

    let now = Utc::now().to_rfc3339();

    // Fill missing required fields
    if !obj.contains_key("id") {
        obj.insert("id".into(), serde_json::Value::String(slug.to_string()));
    }
    if !obj.contains_key("title") {
        // Fall back to slug as title
        obj.insert("title".into(), serde_json::Value::String(slug.to_string()));
    }
    if !obj.contains_key("tags") {
        obj.insert("tags".into(), serde_json::json!([]));
    }
    if !obj.contains_key("entity_count") {
        obj.insert("entity_count".into(), serde_json::json!(0));
    }
    if !obj.contains_key("situation_count") {
        obj.insert("situation_count".into(), serde_json::json!(0));
    }
    if !obj.contains_key("created_at") {
        obj.insert("created_at".into(), serde_json::Value::String(now.clone()));
    }
    if !obj.contains_key("updated_at") {
        obj.insert("updated_at".into(), serde_json::Value::String(now));
    }

    serde_json::from_value(v)
        .map_err(|e| TensaError::Internal(format!("Failed to parse '{path}': {e}")))
}

/// Read entities.json leniently — fills missing required fields with defaults.
fn read_entities_lenient(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    path: &str,
) -> Result<Vec<Entity>> {
    let v = read_json_value(archive, path)?;
    let arr = v
        .as_array()
        .ok_or_else(|| TensaError::Internal(format!("{path}: expected JSON array")))?;

    let now = Utc::now().to_rfc3339();
    let mut entities = Vec::with_capacity(arr.len());

    for (i, item) in arr.iter().enumerate() {
        let mut obj = item.clone();
        if let Some(o) = obj.as_object_mut() {
            if !o.contains_key("id") {
                o.insert(
                    "id".into(),
                    serde_json::Value::String(Uuid::now_v7().to_string()),
                );
            }
            if !o.contains_key("entity_type") {
                o.insert(
                    "entity_type".into(),
                    serde_json::Value::String("Actor".into()),
                );
            }
            if !o.contains_key("properties") {
                o.insert("properties".into(), serde_json::json!({}));
            }
            if !o.contains_key("maturity") {
                o.insert(
                    "maturity".into(),
                    serde_json::Value::String("Candidate".into()),
                );
            }
            if !o.contains_key("confidence") {
                o.insert("confidence".into(), serde_json::json!(1.0));
            }
            if !o.contains_key("provenance") {
                o.insert("provenance".into(), serde_json::json!([]));
            }
            if !o.contains_key("extraction_method") {
                o.insert(
                    "extraction_method".into(),
                    serde_json::Value::String("StructuredImport".into()),
                );
            }
            if !o.contains_key("created_at") {
                o.insert("created_at".into(), serde_json::Value::String(now.clone()));
            }
            if !o.contains_key("updated_at") {
                o.insert("updated_at".into(), serde_json::Value::String(now.clone()));
            }
        }

        match serde_json::from_value::<Entity>(obj) {
            Ok(e) => entities.push(e),
            Err(e) => tracing::warn!("Skipping entity[{i}] in {path}: {e}"),
        }
    }

    Ok(entities)
}

/// Read situations.json leniently — fills missing required fields with defaults.
fn read_situations_lenient(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    path: &str,
) -> Result<Vec<Situation>> {
    let v = read_json_value(archive, path)?;
    let arr = v
        .as_array()
        .ok_or_else(|| TensaError::Internal(format!("{path}: expected JSON array")))?;

    let now = Utc::now().to_rfc3339();
    let mut situations = Vec::with_capacity(arr.len());

    for (i, item) in arr.iter().enumerate() {
        let mut obj = item.clone();
        if let Some(o) = obj.as_object_mut() {
            if !o.contains_key("id") {
                o.insert(
                    "id".into(),
                    serde_json::Value::String(Uuid::now_v7().to_string()),
                );
            }
            if !o.contains_key("temporal") {
                o.insert(
                    "temporal".into(),
                    serde_json::json!({
                        "start": null, "end": null, "granularity": "Unknown", "relations": []
                    }),
                );
            }
            // Fix up temporal sub-fields
            if let Some(t) = o.get_mut("temporal").and_then(|v| v.as_object_mut()) {
                if !t.contains_key("granularity") {
                    t.insert("granularity".into(), serde_json::json!("Unknown"));
                }
                if !t.contains_key("relations") {
                    t.insert("relations".into(), serde_json::json!([]));
                }
            }
            // Fix up spatial sub-fields (precision is required enum)
            if let Some(s) = o.get_mut("spatial").and_then(|v| v.as_object_mut()) {
                if !s.contains_key("precision") {
                    s.insert("precision".into(), serde_json::json!("Approximate"));
                }
            }
            // Fix up game_structure sub-fields
            if let Some(g) = o.get_mut("game_structure").and_then(|v| v.as_object_mut()) {
                if !g.contains_key("game_type") {
                    g.insert("game_type".into(), serde_json::json!({"Custom": "unknown"}));
                }
                if !g.contains_key("info_structure") {
                    g.insert("info_structure".into(), serde_json::json!("Incomplete"));
                }
                if !g.contains_key("maturity") {
                    g.insert("maturity".into(), serde_json::json!("Candidate"));
                }
            }
            if !o.contains_key("raw_content") {
                o.insert("raw_content".into(), serde_json::json!([]));
            }
            if !o.contains_key("narrative_level") {
                o.insert(
                    "narrative_level".into(),
                    serde_json::Value::String("Event".into()),
                );
            }
            if !o.contains_key("maturity") {
                o.insert(
                    "maturity".into(),
                    serde_json::Value::String("Candidate".into()),
                );
            }
            if !o.contains_key("confidence") {
                o.insert("confidence".into(), serde_json::json!(1.0));
            }
            if !o.contains_key("extraction_method") {
                o.insert(
                    "extraction_method".into(),
                    serde_json::Value::String("StructuredImport".into()),
                );
            }
            if !o.contains_key("provenance") {
                o.insert("provenance".into(), serde_json::json!([]));
            }
            if !o.contains_key("causes") {
                o.insert("causes".into(), serde_json::json!([]));
            }
            if !o.contains_key("created_at") {
                o.insert("created_at".into(), serde_json::Value::String(now.clone()));
            }
            if !o.contains_key("updated_at") {
                o.insert("updated_at".into(), serde_json::Value::String(now.clone()));
            }
        }

        match serde_json::from_value::<Situation>(obj) {
            Ok(mut s) => {
                // If no source_span provided, generate one from array position
                // so Timeline ordering works even for hand-crafted archives
                if s.source_span.is_none() {
                    s.source_span = Some(crate::types::SourceSpan {
                        chunk_index: 0,
                        byte_offset_start: 0,
                        byte_offset_end: 0,
                        local_index: i as u16,
                    });
                }
                situations.push(s);
            }
            Err(e) => tracing::warn!("Skipping situation[{i}] in {path}: {e}"),
        }
    }

    Ok(situations)
}

/// Read and deserialize a JSON file from the ZIP archive.
fn read_json_file<T: serde::de::DeserializeOwned>(
    archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
    path: &str,
) -> Result<T> {
    let mut file = archive
        .by_name(path)
        .map_err(|e| TensaError::Internal(format!("Missing archive file '{path}': {e}")))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .map_err(|e| TensaError::Internal(format!("Failed to read '{path}': {e}")))?;
    serde_json::from_str(&buf)
        .map_err(|e| TensaError::Internal(format!("Failed to parse '{path}': {e}")))
}

/// Remap entity references in knowledge facts.
fn remap_knowledge_facts(remap: &HashMap<Uuid, Uuid>, facts: &mut [KnowledgeFact]) {
    for fact in facts {
        if let Some(&new_id) = remap.get(&fact.about_entity) {
            fact.about_entity = new_id;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::archive::export_archive;
    use crate::export::archive_types::ArchiveExportOptions;
    use crate::hypergraph::Hypergraph;
    use crate::narrative::registry::NarrativeRegistry;
    use crate::narrative::types::Narrative;
    use crate::store::memory::MemoryStore;
    use chrono::Utc;
    use std::io::Write;
    use std::sync::Arc;

    fn create_test_archive() -> (Vec<u8>, Arc<MemoryStore>) {
        let store = Arc::new(MemoryStore::new());
        let hg = Hypergraph::new(store.clone());
        let nar_reg = NarrativeRegistry::new(store.clone() as Arc<dyn crate::store::KVStore>);

        let now = Utc::now();
        nar_reg
            .create(Narrative {
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
            })
            .unwrap();

        let alice_id = Uuid::now_v7();
        hg.create_entity(Entity {
            id: alice_id,
            entity_type: EntityType::Actor,
            properties: serde_json::json!({"name": "Alice"}),
            beliefs: None,
            embedding: None,
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
        })
        .unwrap();

        let sit_id = Uuid::now_v7();
        hg.create_situation(Situation {
            id: sit_id,
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
            raw_content: vec![ContentBlock::text("Something happened")],
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
        })
        .unwrap();

        hg.add_participant(Participation {
            entity_id: alice_id,
            situation_id: sit_id,
            role: Role::Protagonist,
            info_set: None,
            action: Some("acts".to_string()),
            payoff: None,
            seq: 0,
        })
        .unwrap();

        let opts = ArchiveExportOptions {
            include_sources: false,
            include_chunks: false,
            include_state_versions: false,
            include_inference: false,
            include_analysis: false,
            include_tuning: false,
            include_embeddings: false,
            include_taxonomy: false,
            include_projects: false,
            include_annotations: false,
            include_pinned_facts: false,
            include_revisions: false,
            include_workshop_reports: false,
            include_narrative_plan: false,
            include_analysis_status: false,
            pretty: true,
            include_synthetic: false,
        };
        let bytes = export_archive(&["test-nar"], &hg, &opts).unwrap();
        (bytes, store)
    }

    #[test]
    fn test_import_archive_roundtrip() {
        let (archive_bytes, _source_store) = create_test_archive();

        // Import into a fresh store
        let target_store = Arc::new(MemoryStore::new());
        let target_hg = Hypergraph::new(target_store.clone());

        let opts = ArchiveImportOptions::default();
        let report = import_archive(&archive_bytes, &target_hg, &opts).unwrap();

        assert_eq!(report.narratives_imported, 1);
        assert_eq!(report.entities_created, 1);
        assert_eq!(report.situations_created, 1);
        assert_eq!(report.participations_created, 1);
        assert!(report.errors.is_empty(), "Errors: {:?}", report.errors);

        // Verify data exists in target
        let entities = target_hg.list_entities_by_narrative("test-nar").unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].properties["name"], "Alice");

        let situations = target_hg.list_situations_by_narrative("test-nar").unwrap();
        assert_eq!(situations.len(), 1);

        let parts = target_hg
            .get_participants_for_situation(&situations[0].id)
            .unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].role, Role::Protagonist);
    }

    #[test]
    fn test_import_archive_merge_mode() {
        let (archive_bytes, _source_store) = create_test_archive();

        // Import once
        let target_store = Arc::new(MemoryStore::new());
        let target_hg = Hypergraph::new(target_store.clone());
        let opts = ArchiveImportOptions {
            merge_mode: true,
            ..Default::default()
        };
        let report1 = import_archive(&archive_bytes, &target_hg, &opts).unwrap();
        assert_eq!(report1.entities_created, 1);

        // Import again in merge mode — should skip existing
        let report2 = import_archive(&archive_bytes, &target_hg, &opts).unwrap();
        assert_eq!(report2.entities_skipped, 1);
        assert_eq!(report2.entities_created, 0);

        // Still only 1 entity
        let entities = target_hg.list_entities_by_narrative("test-nar").unwrap();
        assert_eq!(entities.len(), 1);
    }

    #[test]
    fn test_import_archive_remap_mode() {
        let (archive_bytes, _source_store) = create_test_archive();

        // Import once
        let target_store = Arc::new(MemoryStore::new());
        let target_hg = Hypergraph::new(target_store.clone());
        let opts = ArchiveImportOptions::default();
        let report1 = import_archive(&archive_bytes, &target_hg, &opts).unwrap();
        assert_eq!(report1.entities_created, 1);

        // Import again in remap mode — should create new entity with new UUID
        let report2 = import_archive(&archive_bytes, &target_hg, &opts).unwrap();
        // Narrative gets suffixed
        assert!(!report2.warnings.is_empty());
        // Entity remapped
        assert!(!report2.id_remaps.is_empty() || report2.entities_created == 1);
    }

    #[test]
    fn test_import_archive_lenient_external() {
        // Simulate an archive created by an external tool with minimal fields
        let buf = Vec::new();
        let cursor = std::io::Cursor::new(buf);
        let mut zip = zip::ZipWriter::new(cursor);
        let opts = zip::write::SimpleFileOptions::default();

        let now = Utc::now().to_rfc3339();

        // Manifest
        zip.start_file("manifest.json", opts).unwrap();
        zip.write_all(
            serde_json::json!({
                "tensa_archive_version": "1.0.0",
                "created_at": now,
                "narratives": ["my-book"],
                "layers": {"core": true},
                "strict_references": false
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();

        // narrative.json — MISSING id, entity_count, situation_count, created_at, updated_at
        zip.start_file("narratives/my-book/narrative.json", opts)
            .unwrap();
        zip.write_all(
            serde_json::json!({
                "title": "My Book",
                "genre": "novel"
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();

        // entities.json — MISSING id, maturity, confidence, created_at, etc.
        let eid = Uuid::now_v7().to_string();
        zip.start_file("narratives/my-book/entities.json", opts)
            .unwrap();
        zip.write_all(
            serde_json::json!([
                {"id": eid, "entity_type": "Actor", "properties": {"name": "Alice"}},
                {"entity_type": "Location", "properties": {"name": "The Library"}}
            ])
            .to_string()
            .as_bytes(),
        )
        .unwrap();

        // situations.json — MISSING id, temporal, extraction_method, etc.
        // Second situation has spatial without precision (common external tool omission)
        zip.start_file("narratives/my-book/situations.json", opts)
            .unwrap();
        zip.write_all(
            serde_json::json!([
                {"name": "Alice enters the library", "description": "She walked in quietly"},
                {"name": "Rooftop chase", "description": "Chase across London rooftops",
                 "spatial": {"latitude": 51.5074, "longitude": -0.1278, "description": "London"}}
            ])
            .to_string()
            .as_bytes(),
        )
        .unwrap();

        // participations.json — minimal
        zip.start_file("narratives/my-book/participations.json", opts)
            .unwrap();
        zip.write_all(b"[]").unwrap();

        let cursor = zip.finish().unwrap();
        let archive_bytes = cursor.into_inner();

        // Import
        let target_store = Arc::new(MemoryStore::new());
        let target_hg = Hypergraph::new(target_store.clone());
        let import_opts = ArchiveImportOptions {
            strict_references: false,
            ..Default::default()
        };
        let report = import_archive(&archive_bytes, &target_hg, &import_opts).unwrap();

        assert_eq!(report.narratives_imported, 1);
        assert_eq!(report.entities_created, 2);
        assert_eq!(report.situations_created, 2);
        assert!(report.errors.is_empty(), "Errors: {:?}", report.errors);

        // Verify entities exist with generated UUIDs and defaults
        let entities = target_hg.list_entities_by_narrative("my-book").unwrap();
        assert_eq!(entities.len(), 2);
        let alice = entities
            .iter()
            .find(|e| e.properties["name"] == "Alice")
            .unwrap();
        assert_eq!(alice.confidence, 1.0); // Default
        assert_eq!(alice.maturity, MaturityLevel::Candidate); // Default

        // Location entity should have auto-generated UUID
        let library = entities
            .iter()
            .find(|e| e.properties["name"] == "The Library")
            .unwrap();
        assert_eq!(library.entity_type, EntityType::Location);

        // Situations should exist with defaults
        let situations = target_hg.list_situations_by_narrative("my-book").unwrap();
        assert_eq!(situations.len(), 2);
        let rooftop = situations
            .iter()
            .find(|s| s.name.as_deref() == Some("Rooftop chase"))
            .unwrap();
        assert!(rooftop.spatial.is_some());
        assert_eq!(
            rooftop.spatial.as_ref().unwrap().precision,
            SpatialPrecision::Approximate
        );
    }

    #[test]
    fn test_import_archive_target_narrative_override() {
        let (archive_bytes, _source_store) = create_test_archive();

        let target_store = Arc::new(MemoryStore::new());
        let target_hg = Hypergraph::new(target_store.clone());
        let opts = ArchiveImportOptions {
            target_narrative_id: Some("custom-slug".to_string()),
            ..Default::default()
        };
        let report = import_archive(&archive_bytes, &target_hg, &opts).unwrap();
        assert_eq!(report.narratives_imported, 1);

        // Entity should be in the custom slug
        let entities = target_hg.list_entities_by_narrative("custom-slug").unwrap();
        assert_eq!(entities.len(), 1);
    }

    /// v1.1.0: round-trip the six new layers. Build a narrative with an
    /// annotation, a pinned fact, a narrative plan, and an analysis-status
    /// row, export with all-on options, import into a fresh store, and verify
    /// every record is reachable + the lock state survives.
    #[test]
    fn test_v1_1_layers_roundtrip() {
        use crate::analysis_status::{
            AnalysisSource, AnalysisStatusEntry, AnalysisStatusStore,
        };
        use crate::narrative::continuity;
        use crate::narrative::plan;
        use crate::types::{InferenceJobType, NarrativePlan, PinnedFact};
        use crate::writer::annotation::{create_annotation, Annotation, AnnotationKind};

        let (_seed_bytes, source_store) = create_test_archive();
        let source_hg = Hypergraph::new(source_store.clone());
        let situations = source_hg.list_situations_by_narrative("test-nar").unwrap();
        let sit_id = situations[0].id;
        let now = Utc::now();

        // Annotation pointing at the seeded situation.
        create_annotation(
            &*source_store,
            Annotation {
                id: Uuid::now_v7(),
                situation_id: sit_id,
                kind: AnnotationKind::Comment,
                span: (0, 5),
                body: "needs more pacing".into(),
                source_id: None,
                chunk_id: None,
                author: Some("skill:tensa-narrative-llm".into()),
                detached: false,
                created_at: now,
                updated_at: now,
            },
        )
        .unwrap();

        // Pinned fact (commitment).
        continuity::create_pinned_fact(
            &*source_store,
            PinnedFact {
                id: Uuid::now_v7(),
                narrative_id: "test-nar".into(),
                entity_id: None,
                key: "commitment".into(),
                value: "Chekhov's gun introduced in scene 1".into(),
                note: Some("skill:tensa-narrative-llm".into()),
                created_at: now,
                updated_at: now,
            },
        )
        .unwrap();

        // Narrative plan — only required fields; the rest default.
        plan::upsert_plan(
            &*source_store,
            NarrativePlan {
                narrative_id: "test-nar".into(),
                logline: None,
                synopsis: None,
                premise: Some("Alice navigates the unknown".into()),
                themes: vec!["agency".into()],
                central_conflict: None,
                plot_beats: vec![],
                style: Default::default(),
                length: Default::default(),
                setting: Default::default(),
                notes: String::new(),
                target_audience: None,
                comp_titles: vec![],
                content_warnings: vec![],
                custom: std::collections::HashMap::new(),
                created_at: now,
                updated_at: now,
            },
        )
        .unwrap();

        // Analysis-status row — Skill source, locked.
        let status_store = AnalysisStatusStore::new(source_store.clone());
        status_store
            .upsert(&AnalysisStatusEntry {
                narrative_id: "test-nar".into(),
                job_type: InferenceJobType::ArcClassification,
                scope: "story".into(),
                source: AnalysisSource::Skill,
                skill: Some("tensa-narrative-llm".into()),
                model: Some("claude-opus-4-7".into()),
                completed_at: now,
                locked: true,
                summary: Some("Cinderella arc detected".into()),
                confidence: Some(0.88),
                result_refs: vec![],
            })
            .unwrap();

        // Export with all v1.1.0 layers ON (default).
        let opts = ArchiveExportOptions::default();
        let bytes = export_archive(&["test-nar"], &source_hg, &opts).unwrap();

        // Import into a fresh store.
        let target_store = Arc::new(MemoryStore::new());
        let target_hg = Hypergraph::new(target_store.clone());
        let report =
            import_archive(&bytes, &target_hg, &ArchiveImportOptions::default()).unwrap();

        // The four new counters should each be 1.
        assert_eq!(
            report.annotations_created, 1,
            "annotations_created (errors: {:?}, warnings: {:?})",
            report.errors, report.warnings
        );
        assert_eq!(report.pinned_facts_created, 1);
        assert_eq!(report.narrative_plans_created, 1);
        assert_eq!(report.analysis_status_entries_created, 1);

        // Annotation body must round-trip verbatim.
        let target_situations = target_hg.list_situations_by_narrative("test-nar").unwrap();
        let target_sit_id = target_situations[0].id;
        let target_anns =
            crate::writer::annotation::list_annotations_for_situation(&*target_store, &target_sit_id)
                .unwrap();
        assert_eq!(target_anns.len(), 1);
        assert_eq!(target_anns[0].body, "needs more pacing");
        assert_eq!(
            target_anns[0].author.as_deref(),
            Some("skill:tensa-narrative-llm")
        );

        // Pinned fact key/value must round-trip verbatim.
        let target_facts =
            crate::narrative::continuity::list_pinned_facts(&*target_store, "test-nar").unwrap();
        assert_eq!(target_facts.len(), 1);
        assert_eq!(target_facts[0].key, "commitment");
        assert_eq!(
            target_facts[0].value,
            "Chekhov's gun introduced in scene 1"
        );

        // Narrative plan premise must round-trip.
        let target_plan = crate::narrative::plan::get_plan(&*target_store, "test-nar")
            .unwrap()
            .expect("narrative plan should round-trip");
        assert_eq!(
            target_plan.premise.as_deref(),
            Some("Alice navigates the unknown")
        );
        assert_eq!(target_plan.themes, vec!["agency".to_string()]);

        // Lock state must survive — this is the load-bearing property: a
        // re-imported archive with a Skill+locked row should still report
        // locked=true so subsequent bulk-analysis runs skip it.
        let target_status = AnalysisStatusStore::new(target_store);
        let row = target_status
            .get("test-nar", &InferenceJobType::ArcClassification, "story")
            .unwrap()
            .expect("analysis-status row should round-trip");
        assert_eq!(row.source, AnalysisSource::Skill);
        assert!(row.locked, "lock state must survive export/import");
        assert_eq!(row.skill.as_deref(), Some("tensa-narrative-llm"));
    }
}
