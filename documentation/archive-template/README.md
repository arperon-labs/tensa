# Tensa Narrative Archive (.tensa) Template

This directory is a template for building a `.tensa` archive — the lossless exchange format for [TENSA](https://github.com/arperon-labs/tensa). ZIP this directory (or use any ZIP library) and rename to `.tensa`.

**Format version: 1.1.0** (TENSA 0.79.5+). v1.1.0 added six new optional layers — `annotations`, `pinned_facts`, `revisions`, `workshop_reports`, `narrative_plan`, `analysis_status` — needed for `/tensa-narrative-llm` skill output to round-trip losslessly. Older v1.0.0 archives import cleanly into v1.1.0; v1.1.0 archives import into older readers minus the new layers (every new layer flag uses `#[serde(default)]` and the underlying directories are simply ignored).

## Quick Start (Minimal Archive)

You only need **4 files** to create a valid archive:

```
manifest.json                              <-- required
narratives/{slug}/
  narrative.json                           <-- required
  entities.json                            <-- required (can be empty [])
  situations.json                          <-- required (can be empty [])
  participations.json                      <-- required (can be empty [])
```

Everything else is optional. Set `"strict_references": false` in the manifest if your entities/situations don't have UUIDs yet — Tensa will generate them on import.

## How to Create a .tensa File

### From Python

```python
import json, zipfile, io, datetime, uuid

def new_id():
    """Generate a UUIDv7-like ID (time-ordered)."""
    return str(uuid.uuid4())  # v4 works too, Tensa accepts any valid UUID

buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
    # 1. Write manifest
    zf.writestr("manifest.json", json.dumps({
        "tensa_archive_version": "1.1.0",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "created_by": {"tool": "my-script", "version": "1.0"},
        "narratives": ["my-narrative"],
        "layers": {"core": True},
        "strict_references": False
    }, indent=2))
    
    # 2. Write narrative metadata
    zf.writestr("narratives/my-narrative/narrative.json", json.dumps({
        "id": "my-narrative",
        "title": "My Narrative",
        "genre": "novel",
        "tags": ["fiction"],
        "entity_count": 0,
        "situation_count": 0,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }, indent=2))
    
    # 3. Entities
    alice_id = new_id()
    bob_id = new_id()
    zf.writestr("narratives/my-narrative/entities.json", json.dumps([
        {"id": alice_id, "entity_type": "Actor", "properties": {"name": "Alice"}, "confidence": 1.0},
        {"id": bob_id, "entity_type": "Actor", "properties": {"name": "Bob"}, "confidence": 1.0},
    ], indent=2))
    
    # 4. Situations
    sit1_id = new_id()
    zf.writestr("narratives/my-narrative/situations.json", json.dumps([
        {"id": sit1_id, "name": "The Meeting", "description": "Alice meets Bob at the cafe",
         "narrative_level": "Scene", "confidence": 1.0,
         "raw_content": [{"content_type": "Text", "content": "Alice walked into the cafe..."}]}
    ], indent=2))
    
    # 5. Participations
    zf.writestr("narratives/my-narrative/participations.json", json.dumps([
        {"entity_id": alice_id, "situation_id": sit1_id, "role": "Protagonist", "action": "enters cafe"},
        {"entity_id": bob_id, "situation_id": sit1_id, "role": "Witness"},
    ], indent=2))

with open("my-narrative.tensa", "wb") as f:
    f.write(buf.getvalue())
```

### From an LLM Agent

If you are an LLM agent building a .tensa archive from a book or document:

1. Read the source text
2. Identify **entities** (people, places, organizations, objects, concepts)
3. Identify **situations** (events, scenes, meetings, decisions, actions)
4. Identify **participations** (who was involved in what, and their role)
5. Identify **causal links** (what caused what)
6. Fill in the JSON templates below
7. ZIP the directory and output as `.tensa`

---

## Directory Structure

```
archive.tensa (ZIP)
|
+-- manifest.json                           REQUIRED
+-- narratives/
|   +-- {slug}/
|   |   +-- narrative.json                  REQUIRED
|   |   +-- entities.json                   REQUIRED
|   |   +-- situations.json                 REQUIRED
|   |   +-- participations.json             REQUIRED
|   |   +-- causal_links.json               optional
|   |   +-- sources/
|   |   |   +-- sources.json                optional
|   |   |   +-- attributions.json           optional
|   |   |   +-- contentions.json            optional
|   |   +-- chunks/
|   |   |   +-- chunks.json                 optional
|   |   +-- state_versions/
|   |   |   +-- state_versions.json         optional
|   |   +-- inference/
|   |   |   +-- results.json                optional
|   |   +-- analysis/
|   |   |   +-- communities.json            optional
|   |   +-- tuning/
|   |   |   +-- tuned_prompt.json           optional
|   |   +-- embeddings/
|   |   |   +-- entity_embeddings.bin        optional
|   |   |   +-- situation_embeddings.bin     optional
|   |   |   +-- embedding_index.json         optional
|   |   +-- annotations/                     v1.1.0
|   |   |   +-- annotations.json             optional   <-- /tensa-narrative-llm: dramatic-irony, subplots, arc-classification, …
|   |   +-- pinned_facts/                    v1.1.0
|   |   |   +-- pinned_facts.json            optional   <-- /tensa-narrative-llm: commitments
|   |   +-- revisions/                       v1.1.0
|   |   |   +-- revisions.json               optional   <-- /tensa-narrative-llm: narrative-diagnose-and-repair
|   |   +-- workshop_reports/                v1.1.0
|   |   |   +-- workshop_reports.json        optional   <-- run_workshop output
|   |   +-- plan/                            v1.1.0
|   |   |   +-- narrative_plan.json          optional   <-- writer-flow narrative plan
|   |   +-- analysis_status/                 v1.1.0
|   |       +-- entries.json                 optional   <-- preserves Source: Skill + locked: true rows; bulk-analysis won't overwrite skill output after re-import
|   +-- {another-slug}/
|       +-- ...
+-- taxonomy/
|   +-- taxonomy.json                        optional
+-- projects/
    +-- {slug}.json                          optional
```

### v1.1.0 Layers — Why They Matter

The six layers added in v1.1.0 carry **non-regenerable content** that lives outside the entity/situation graph:

| Layer | What It Stores | Source |
|---|---|---|
| `annotations` | Inline comments, footnotes, citations on situation prose. Byte-span anchored. | Reviewer / `/tensa-narrative-llm` skill |
| `pinned_facts` | Continuity facts: entity property pins (e.g. "Alice is 23") + narrative-wide rules (e.g. Chekhov's guns) | `/tensa-narrative-llm commitments`, manual writer entries |
| `revisions` | Git-like snapshots of the narrative with author + message + parent chain | `commit_narrative_revision` (called by `narrative-diagnose-and-repair`) |
| `workshop_reports` | Three-tier critique reports (Junior / Mid / Senior) with structured findings | `run_workshop` |
| `narrative_plan` | Writer doc — logline / synopsis / premise / themes / plot beats / style targets | Writer flow |
| `analysis_status` | Per-narrative registry of which inference jobs ran, by what source (Auto / Skill / Manual), and whether the result is **locked** against bulk-analysis overwrite | TENSA worker pool + `/tensa-narrative-llm` skill |

**The load-bearing layer is `analysis_status`.** Without it, a re-imported archive loses the `Source: Skill, locked: true` rows that mark a result as skill-attested. The next bulk-analysis run on the imported narrative would silently re-run the cheap-LLM server worker over those job types and overwrite the skill output. Including this layer makes the lock survive the round-trip.

---

## File Schemas

See the example JSON files in this template for complete field-level documentation. Each file has every field annotated with its type, whether it's required, and what values are valid.

### Import Behavior

When Tensa imports an archive:

- **Missing optional fields** get safe defaults (confidence=1.0, maturity="Candidate", timestamps=now)
- **Missing UUIDs** are auto-generated (set `strict_references: false` in manifest)
- **UUID clashes** are resolved by generating new UUIDs and remapping all references
- **Narrative slug clashes** append a numeric suffix (`my-narrative-2`)
- **Merge mode** (opt-in): skips records whose UUIDs already exist

### Version Compatibility

- Format version is semver: `1.x.x`
- Readers ignore unknown JSON fields (forward-compatible)
- Readers ignore unknown directories/files in the ZIP
- Major version bump = breaking change (readers must reject)
- Minor version bump = new optional features (readers must ignore gracefully)
