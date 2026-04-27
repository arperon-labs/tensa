//! Sprint W13 — Compile profiles + epub / docx.
//!
//! A `CompileProfile` captures saved compile rules: which scenes to
//! include (by label / status), per-level heading template, front/back matter,
//! footnote style, and whether annotations should render. The compile
//! dispatcher resolves the profile into a Markdown intermediate (reusing
//! `export_manuscript`), then emits Markdown / epub / docx.
//!
//! epub output is hand-rolled: a minimal EPUB 3 container zipped by the
//! already-bundled `zip` crate. Everything derives from a single Markdown
//! intermediate so output parity is guaranteed.

use std::io::{Cursor, Write};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zip::write::FileOptions;

use crate::error::{Result, TensaError};
use crate::export::NarrativeExport;
use crate::store::KVStore;
use crate::types::{ContentType, NarrativeLevel, Situation};
use crate::writer::annotation::{
    list_annotations_for_scenes, render_annotation_parts, Annotation, AnnotationKind,
};

// ─── Profile types ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileProfile {
    #[serde(default)]
    pub id: Uuid,
    #[serde(default)]
    pub narrative_id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Only include situations whose `label` is in this set (empty = include all).
    #[serde(default)]
    pub include_labels: Vec<String>,
    /// Exclude situations whose `label` is in this set.
    #[serde(default)]
    pub exclude_labels: Vec<String>,
    /// Only include situations whose `status` is in this set (empty = include all).
    #[serde(default)]
    pub include_statuses: Vec<String>,
    /// Per-level heading template. Supported placeholders: `{name}`, `{index}`.
    /// Default: `"# {name}"` for Story, `"## {name}"` for Arc, etc.
    #[serde(default)]
    pub heading_templates: Vec<HeadingRule>,
    /// Markdown to prepend as front matter (e.g. title page).
    #[serde(default)]
    pub front_matter_md: Option<String>,
    /// Markdown to append as back matter (e.g. author bio).
    #[serde(default)]
    pub back_matter_md: Option<String>,
    #[serde(default)]
    pub footnote_style: FootnoteStyle,
    #[serde(default)]
    pub include_comments: bool,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadingRule {
    pub level: NarrativeLevel,
    pub template: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FootnoteStyle {
    Inline,
    Endnotes,
    None,
}
impl Default for FootnoteStyle {
    fn default() -> Self {
        FootnoteStyle::Endnotes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompileFormat {
    Markdown,
    Epub,
    Docx,
}

// ─── KV persistence ──────────────────────────────────────────

pub(crate) fn profile_key(id: &Uuid) -> Vec<u8> {
    let mut k = b"compile/".to_vec();
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn narrative_index_key(narrative_id: &str, id: &Uuid) -> Vec<u8> {
    let mut k = format!("compile/n/{}/", narrative_id).into_bytes();
    k.extend_from_slice(id.as_bytes());
    k
}

pub(crate) fn narrative_index_prefix(narrative_id: &str) -> Vec<u8> {
    format!("compile/n/{}/", narrative_id).into_bytes()
}

pub fn create_profile(store: &dyn KVStore, mut profile: CompileProfile) -> Result<CompileProfile> {
    if profile.name.trim().is_empty() {
        return Err(TensaError::InvalidQuery(
            "compile profile name cannot be empty".into(),
        ));
    }
    let now = Utc::now();
    if profile.id.is_nil() {
        profile.id = Uuid::now_v7();
    }
    profile.created_at = now;
    profile.updated_at = now;
    let bytes = serde_json::to_vec(&profile)?;
    store.put(&profile_key(&profile.id), &bytes)?;
    store.put(
        &narrative_index_key(&profile.narrative_id, &profile.id),
        &[],
    )?;
    Ok(profile)
}

pub fn get_profile(store: &dyn KVStore, id: &Uuid) -> Result<CompileProfile> {
    match store.get(&profile_key(id))? {
        Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
        None => Err(TensaError::QueryError(format!(
            "compile profile {id} not found"
        ))),
    }
}

pub fn list_profiles_for_narrative(
    store: &dyn KVStore,
    narrative_id: &str,
) -> Result<Vec<CompileProfile>> {
    let prefix = narrative_index_prefix(narrative_id);
    let mut out = crate::store::scan_uuid_index(store, &prefix, |id| get_profile(store, id))?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn delete_profile(store: &dyn KVStore, id: &Uuid) -> Result<()> {
    let profile = get_profile(store, id)?;
    store.delete(&profile_key(id))?;
    store.delete(&narrative_index_key(&profile.narrative_id, id))?;
    Ok(())
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct ProfilePatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub include_labels: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_labels: Option<Vec<String>>,
    #[serde(default)]
    pub include_statuses: Option<Vec<String>>,
    #[serde(default)]
    pub heading_templates: Option<Vec<HeadingRule>>,
    #[serde(default)]
    pub front_matter_md: Option<Option<String>>,
    #[serde(default)]
    pub back_matter_md: Option<Option<String>>,
    #[serde(default)]
    pub footnote_style: Option<FootnoteStyle>,
    #[serde(default)]
    pub include_comments: Option<bool>,
}

pub fn update_profile(
    store: &dyn KVStore,
    id: &Uuid,
    patch: ProfilePatch,
) -> Result<CompileProfile> {
    let mut p = get_profile(store, id)?;
    if let Some(v) = patch.name {
        p.name = v;
    }
    if let Some(v) = patch.description {
        p.description = v;
    }
    if let Some(v) = patch.include_labels {
        p.include_labels = v;
    }
    if let Some(v) = patch.exclude_labels {
        p.exclude_labels = v;
    }
    if let Some(v) = patch.include_statuses {
        p.include_statuses = v;
    }
    if let Some(v) = patch.heading_templates {
        p.heading_templates = v;
    }
    if let Some(v) = patch.front_matter_md {
        p.front_matter_md = v;
    }
    if let Some(v) = patch.back_matter_md {
        p.back_matter_md = v;
    }
    if let Some(v) = patch.footnote_style {
        p.footnote_style = v;
    }
    if let Some(v) = patch.include_comments {
        p.include_comments = v;
    }
    p.updated_at = Utc::now();
    let bytes = serde_json::to_vec(&p)?;
    store.put(&profile_key(id), &bytes)?;
    Ok(p)
}

// ─── Compile dispatch ────────────────────────────────────────

/// Render a narrative through a profile, producing the bytes for the target
/// format. The markdown intermediate is the canonical representation.
pub fn compile(
    store: &dyn KVStore,
    data: &NarrativeExport,
    profile: &CompileProfile,
    format: CompileFormat,
) -> Result<Vec<u8>> {
    let md = render_markdown(store, data, profile)?;
    match format {
        CompileFormat::Markdown => Ok(md.into_bytes()),
        CompileFormat::Epub => render_epub(&md, profile),
        CompileFormat::Docx => render_docx(&md),
    }
}

fn render_markdown(
    store: &dyn KVStore,
    data: &NarrativeExport,
    profile: &CompileProfile,
) -> Result<String> {
    let mut out = String::with_capacity(16 * 1024);

    // Front matter
    if let Some(fm) = &profile.front_matter_md {
        out.push_str(fm);
        out.push_str("\n\n");
    }

    out.push_str(&format!("# {}\n\n", profile.name));

    // Filter + sort situations.
    let filtered: Vec<&Situation> = data
        .situations
        .iter()
        .filter(|s| profile_accepts(profile, s))
        .collect();
    let mut sorted = filtered.clone();
    sorted.sort_by_key(|s| crate::writer::scene::manuscript_sort_key(s));

    // Pre-load annotations once — avoids the N+1 list_annotations_for_situation
    // pattern when rendering many scenes.
    let scene_ids: std::collections::HashSet<Uuid> = sorted.iter().map(|s| s.id).collect();
    let annotations_by_scene = list_annotations_for_scenes(store, &scene_ids).unwrap_or_default();

    // Collect endnote material if the profile wants Endnotes style.
    let mut endnote_markers: Vec<(Uuid, Vec<(usize, String)>)> = Vec::new();
    let mut endnote_block = String::new();

    for s in &sorted {
        let heading = resolve_heading(profile, s.narrative_level, s.name.as_deref());
        if !heading.is_empty() {
            out.push_str(&heading);
            out.push_str("\n\n");
        }
        let mut scene_prose = String::new();
        for block in &s.raw_content {
            if matches!(
                block.content_type,
                ContentType::Text | ContentType::Dialogue
            ) {
                scene_prose.push_str(&block.content);
                scene_prose.push_str("\n\n");
            }
        }

        // Annotations on this scene (pre-loaded above).
        let mut anns = annotations_by_scene.get(&s.id).cloned().unwrap_or_default();
        anns.retain(|a| !a.detached);
        let (markers_block, bib_fragment) = render_annotations_segment(&anns, profile);
        if profile.footnote_style == FootnoteStyle::Endnotes && !markers_block.is_empty() {
            endnote_markers.push((s.id, markers_block));
            endnote_block.push_str(&bib_fragment);
            out.push_str(&scene_prose);
        } else if profile.footnote_style == FootnoteStyle::Inline && !markers_block.is_empty() {
            out.push_str(&apply_inline_markers(&scene_prose, &markers_block));
            out.push('\n');
            out.push_str(&bib_fragment);
            out.push('\n');
        } else {
            out.push_str(&scene_prose);
        }

        // Comments appear only if include_comments=true (editor review compiles).
        if profile.include_comments {
            for a in &anns {
                if a.kind == AnnotationKind::Comment {
                    out.push_str(&format!("> COMMENT: {}\n\n", a.body));
                }
            }
        }
    }

    if !endnote_block.is_empty() && profile.footnote_style == FootnoteStyle::Endnotes {
        out.push_str("\n\n---\n\n## Endnotes\n\n");
        out.push_str(&endnote_block);
    }

    if let Some(bm) = &profile.back_matter_md {
        out.push_str("\n\n---\n\n");
        out.push_str(bm);
    }

    Ok(out)
}

fn profile_accepts(p: &CompileProfile, s: &Situation) -> bool {
    if !p.include_labels.is_empty() {
        match &s.label {
            Some(l) if p.include_labels.contains(l) => {}
            _ => return false,
        }
    }
    if let Some(l) = &s.label {
        if p.exclude_labels.contains(l) {
            return false;
        }
    }
    if !p.include_statuses.is_empty() {
        match &s.status {
            Some(st) if p.include_statuses.contains(st) => {}
            _ => return false,
        }
    }
    true
}

fn resolve_heading(profile: &CompileProfile, level: NarrativeLevel, name: Option<&str>) -> String {
    let template = profile
        .heading_templates
        .iter()
        .find(|h| h.level == level)
        .map(|h| h.template.clone())
        .unwrap_or_else(|| default_heading_template(level));
    let n = name.unwrap_or("Untitled");
    template.replace("{name}", n)
}

fn default_heading_template(level: NarrativeLevel) -> String {
    match level {
        NarrativeLevel::Story => "# {name}".into(),
        NarrativeLevel::Arc => "## {name}".into(),
        NarrativeLevel::Sequence => "### {name}".into(),
        NarrativeLevel::Scene => "#### {name}".into(),
        NarrativeLevel::Beat => String::new(),
        NarrativeLevel::Event => String::new(),
    }
}

fn render_annotations_segment(
    anns: &[Annotation],
    profile: &CompileProfile,
) -> (Vec<(usize, String)>, String) {
    let include_footnotes = profile.footnote_style != FootnoteStyle::None;
    let (markers, footnote_body, bib_body) = render_annotation_parts(anns, include_footnotes);
    let mut combined = footnote_body;
    combined.push_str(&bib_body);
    (markers, combined)
}

fn apply_inline_markers(prose: &str, markers: &[(usize, String)]) -> String {
    let mut sorted = markers.to_vec();
    sorted.sort_by_key(|(pos, _)| *pos);
    let mut out = String::with_capacity(prose.len() + 16 * sorted.len());
    let mut cursor = 0;
    for (pos, m) in sorted {
        let pos = pos.min(prose.len());
        out.push_str(&prose[cursor..pos]);
        out.push_str(&m);
        cursor = pos;
    }
    out.push_str(&prose[cursor..]);
    out
}

// ─── epub ────────────────────────────────────────────────────

/// Build a minimal EPUB 3 from a Markdown body. Wraps the body in a single
/// XHTML document so readers render it directly.
fn render_epub(md: &str, profile: &CompileProfile) -> Result<Vec<u8>> {
    let buf = Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(buf);

    // mimetype must be first, stored (no compression).
    zip.start_file(
        "mimetype",
        FileOptions::<()>::default().compression_method(zip::CompressionMethod::Stored),
    )
    .map_err(|e| TensaError::QueryError(format!("epub zip: {e}")))?;
    zip.write_all(b"application/epub+zip")
        .map_err(|e| TensaError::QueryError(format!("epub write: {e}")))?;

    let opts = FileOptions::<()>::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("META-INF/container.xml", opts)
        .map_err(|e| TensaError::QueryError(format!("epub zip: {e}")))?;
    zip.write_all(CONTAINER_XML.as_bytes())
        .map_err(|e| TensaError::QueryError(format!("epub write: {e}")))?;

    let book_id = Uuid::now_v7();
    let title = html_escape(&profile.name);
    let body_html = markdown_to_xhtml(md);

    let content_opf = CONTENT_OPF
        .replace("{book_id}", &book_id.to_string())
        .replace("{title}", &title);
    zip.start_file("OEBPS/content.opf", opts)
        .map_err(|e| TensaError::QueryError(format!("epub zip: {e}")))?;
    zip.write_all(content_opf.as_bytes())
        .map_err(|e| TensaError::QueryError(format!("epub write: {e}")))?;

    let nav_xhtml = NAV_XHTML.replace("{title}", &title);
    zip.start_file("OEBPS/nav.xhtml", opts)
        .map_err(|e| TensaError::QueryError(format!("epub zip: {e}")))?;
    zip.write_all(nav_xhtml.as_bytes())
        .map_err(|e| TensaError::QueryError(format!("epub write: {e}")))?;

    let chapter_xhtml = CHAPTER_XHTML
        .replace("{title}", &title)
        .replace("{body}", &body_html);
    zip.start_file("OEBPS/chapter.xhtml", opts)
        .map_err(|e| TensaError::QueryError(format!("epub zip: {e}")))?;
    zip.write_all(chapter_xhtml.as_bytes())
        .map_err(|e| TensaError::QueryError(format!("epub write: {e}")))?;

    let out = zip
        .finish()
        .map_err(|e| TensaError::QueryError(format!("epub finish: {e}")))?;
    Ok(out.into_inner())
}

const CONTAINER_XML: &str = r#"<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"#;

const CONTENT_OPF: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">urn:uuid:{book_id}</dc:identifier>
    <dc:title>{title}</dc:title>
    <dc:language>en</dc:language>
    <meta property="dcterms:modified">2026-04-19T00:00:00Z</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter"/>
  </spine>
</package>"#;

const NAV_XHTML: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>{title}</title></head>
  <body>
    <nav epub:type="toc"><h1>Table of Contents</h1>
      <ol><li><a href="chapter.xhtml">{title}</a></li></ol>
    </nav>
  </body>
</html>"#;

const CHAPTER_XHTML: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>{title}</title></head>
  <body>{body}</body>
</html>"#;

/// Convert our Markdown dialect to minimal XHTML. We intentionally skip heavy
/// parsing: headings → `<h1..6>`, blank-line paragraphs, `---` → `<hr/>`.
fn markdown_to_xhtml(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + 128);
    for block in md.split("\n\n") {
        let trimmed = block.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "---" {
            out.push_str("<hr/>");
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#### ") {
            out.push_str(&format!("<h4>{}</h4>", html_escape(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("### ") {
            out.push_str(&format!("<h3>{}</h3>", html_escape(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("## ") {
            out.push_str(&format!("<h2>{}</h2>", html_escape(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("# ") {
            out.push_str(&format!("<h1>{}</h1>", html_escape(rest)));
        } else {
            out.push_str(&format!(
                "<p>{}</p>",
                html_escape(trimmed).replace('\n', "<br/>")
            ));
        }
    }
    out
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

// ─── docx ────────────────────────────────────────────────────

/// docx-rs is behind the `docparse` feature. If not enabled, return a stub
/// error so callers get a clear message instead of a runtime panic.
#[cfg(feature = "docparse")]
fn render_docx(md: &str) -> Result<Vec<u8>> {
    use docx_rs::*;
    let mut doc = Docx::new();
    for block in md.split("\n\n") {
        let trimmed = block.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let (text, style) = if let Some(rest) = trimmed.strip_prefix("#### ") {
            (rest, Some("Heading4"))
        } else if let Some(rest) = trimmed.strip_prefix("### ") {
            (rest, Some("Heading3"))
        } else if let Some(rest) = trimmed.strip_prefix("## ") {
            (rest, Some("Heading2"))
        } else if let Some(rest) = trimmed.strip_prefix("# ") {
            (rest, Some("Heading1"))
        } else {
            (trimmed, None)
        };
        let mut para = Paragraph::new().add_run(Run::new().add_text(text));
        if let Some(s) = style {
            para = para.style(s);
        }
        doc = doc.add_paragraph(para);
    }
    // docx-rs's XMLDocx::pack requires Write + Seek; wrap the Vec in a Cursor.
    let mut cursor = Cursor::new(Vec::new());
    doc.build()
        .pack(&mut cursor)
        .map_err(|e| TensaError::QueryError(format!("docx: {e}")))?;
    Ok(cursor.into_inner())
}

#[cfg(not(feature = "docparse"))]
fn render_docx(_md: &str) -> Result<Vec<u8>> {
    Err(TensaError::QueryError(
        "docx output requires the `docparse` feature to be enabled".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::MemoryStore;
    use crate::types::{
        AllenInterval, ContentBlock, ExtractionMethod, MaturityLevel, NarrativeLevel,
        TimeGranularity,
    };
    use std::sync::Arc;

    fn make_scene(
        narrative: &str,
        name: &str,
        level: NarrativeLevel,
        prose: &str,
        label: Option<&str>,
        status: Option<&str>,
        order: Option<u32>,
    ) -> Situation {
        Situation {
            id: Uuid::now_v7(),
            properties: serde_json::Value::Null,
            name: Some(name.into()),
            description: None,
            temporal: AllenInterval {
                start: None,
                end: None,
                granularity: TimeGranularity::Unknown,
                relations: vec![],
                fuzzy_endpoints: None,
            },
            spatial: None,
            game_structure: None,
            causes: vec![],
            deterministic: None,
            probabilistic: None,
            embedding: None,
            raw_content: vec![ContentBlock::text(prose)],
            narrative_level: level,
            discourse: None,
            maturity: MaturityLevel::Candidate,
            confidence: 0.9,
            confidence_breakdown: None,
            extraction_method: ExtractionMethod::HumanEntered,
            provenance: vec![],
            narrative_id: Some(narrative.into()),
            source_chunk_id: None,
            source_span: None,
            synopsis: None,
            manuscript_order: order,
            parent_situation_id: None,
            label: label.map(|s| s.into()),
            status: status.map(|s| s.into()),
            keywords: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            deleted_at: None,
            transaction_time: None,
        }
    }

    fn default_profile(narrative: &str) -> CompileProfile {
        CompileProfile {
            id: Uuid::nil(),
            narrative_id: narrative.into(),
            name: "Novel".into(),
            description: None,
            include_labels: vec![],
            exclude_labels: vec![],
            include_statuses: vec![],
            heading_templates: vec![],
            front_matter_md: None,
            back_matter_md: None,
            footnote_style: FootnoteStyle::Endnotes,
            include_comments: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn narrative_export(situations: Vec<Situation>) -> NarrativeExport {
        NarrativeExport {
            narrative_id: situations
                .first()
                .and_then(|s| s.narrative_id.clone())
                .unwrap_or_default(),
            entities: vec![],
            situations,
            participations: vec![],
            causal_links: vec![],
        }
    }

    #[test]
    fn test_crud_profile() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let p = create_profile(&*store, default_profile("n1")).unwrap();
        let list = list_profiles_for_narrative(&*store, "n1").unwrap();
        assert_eq!(list.len(), 1);
        let got = get_profile(&*store, &p.id).unwrap();
        assert_eq!(got.name, "Novel");
        delete_profile(&*store, &p.id).unwrap();
        assert!(list_profiles_for_narrative(&*store, "n1")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_markdown_respects_include_labels() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let a = make_scene(
            "n1",
            "A",
            NarrativeLevel::Scene,
            "Alpha prose.",
            Some("final"),
            None,
            Some(1000),
        );
        let b = make_scene(
            "n1",
            "B",
            NarrativeLevel::Scene,
            "Bravo prose.",
            Some("draft"),
            None,
            Some(2000),
        );
        let data = narrative_export(vec![a, b]);
        let mut p = default_profile("n1");
        p.include_labels = vec!["final".into()];
        let bytes = compile(&*store, &data, &p, CompileFormat::Markdown).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.contains("Alpha prose."));
        assert!(!s.contains("Bravo prose."));
    }

    #[test]
    fn test_markdown_respects_manuscript_order() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let a = make_scene(
            "n1",
            "A",
            NarrativeLevel::Scene,
            "Alpha prose.",
            None,
            None,
            Some(2000),
        );
        let b = make_scene(
            "n1",
            "B",
            NarrativeLevel::Scene,
            "Bravo prose.",
            None,
            None,
            Some(1000),
        );
        let data = narrative_export(vec![a, b]);
        let p = default_profile("n1");
        let bytes = compile(&*store, &data, &p, CompileFormat::Markdown).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        let a_pos = s.find("Alpha").unwrap();
        let b_pos = s.find("Bravo").unwrap();
        assert!(b_pos < a_pos);
    }

    #[test]
    fn test_front_and_back_matter_splice() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let a = make_scene(
            "n1",
            "A",
            NarrativeLevel::Scene,
            "Body.",
            None,
            None,
            Some(1000),
        );
        let data = narrative_export(vec![a]);
        let mut p = default_profile("n1");
        p.front_matter_md = Some("*FRONTMATTER*".into());
        p.back_matter_md = Some("*BACKMATTER*".into());
        let bytes = compile(&*store, &data, &p, CompileFormat::Markdown).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        assert!(s.starts_with("*FRONTMATTER*"));
        assert!(s.trim_end().ends_with("*BACKMATTER*"));
    }

    #[test]
    fn test_epub_has_mimetype_and_container() {
        let store: Arc<dyn KVStore> = Arc::new(MemoryStore::new());
        let a = make_scene(
            "n1",
            "A",
            NarrativeLevel::Scene,
            "Body.",
            None,
            None,
            Some(1000),
        );
        let data = narrative_export(vec![a]);
        let p = default_profile("n1");
        let bytes = compile(&*store, &data, &p, CompileFormat::Epub).unwrap();
        // zip magic
        assert_eq!(&bytes[0..2], b"PK");
        // Validate we can unzip and find required entries.
        let reader = std::io::Cursor::new(&bytes);
        let mut zip = zip::ZipArchive::new(reader).unwrap();
        assert!(zip.by_name("mimetype").is_ok());
        assert!(zip.by_name("META-INF/container.xml").is_ok());
        assert!(zip.by_name("OEBPS/content.opf").is_ok());
        assert!(zip.by_name("OEBPS/chapter.xhtml").is_ok());
    }

    #[test]
    fn test_markdown_to_xhtml_converts_headings() {
        let md = "# Title\n\nBody paragraph.\n\n---\n\n## Subsection";
        let x = markdown_to_xhtml(md);
        assert!(x.contains("<h1>Title</h1>"));
        assert!(x.contains("<p>Body paragraph.</p>"));
        assert!(x.contains("<hr/>"));
        assert!(x.contains("<h2>Subsection</h2>"));
    }
}
