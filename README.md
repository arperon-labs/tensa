# TENSA — Temporal Narrative Tensor Architecture

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](license/LICENSE)
[![Commercial license available](https://img.shields.io/badge/commercial%20license-available-green.svg)](license/COMMERCIAL-LICENSE-TERMS.md)
[![Rust](https://img.shields.io/badge/rust-1.83%2B-orange.svg)](https://www.rust-lang.org/)

**TENSA is a multi-fidelity narrative storage, reasoning, and inference engine.** It represents multi-actor event systems — novels, criminal cases, intelligence corpora, disinformation operations, simulated wargames — as a **temporal hypergraph** with mixed deterministic-and-latent annotations, queryable through a custom DSL called **TensaQL**.

It is, in plain language: a graph database for *stories*, with reasoning operators (causal, game-theoretic, fuzzy, argumentation-based, opinion-dynamic) that compose over the same substrate.

---

## What problem does TENSA solve?

Most graph databases assume facts are crisp and edges are binary. Most narrative tools assume the writer hands them clean structure. Real multi-actor event systems are neither:

- **Facts have confidence**, not just presence. A claim is sourced, contradicted, partially refuted, or pinned by an analyst.
- **Time is fuzzy**. "Before the meeting" or "shortly after the explosion" is the user's actual mental model — not a timestamp.
- **Hyperedges, not edges**. A scene/incident/situation involves *N* participants in *N* roles simultaneously, with their own information sets and payoffs.
- **Maturity matters**. A candidate observation, a corroborated fact, and a canon-pinned ground truth need separate lanes.

TENSA gives you all of that with a queryable foundation: hypergraph + Allen interval algebra + confidence pipeline + 178 MCP tools + a chat-driven Studio UI (open-core; ships separately).

---

## Architecture at a glance

```
Layer 5: API          (REST endpoints — axum, feature-gated)
Layer 4: Query        (TensaQL parser + planner + executor)
Layer 3c: Synth       (surrogate-model generation — EATH)
Layer 3b: Narrative   (cross-narrative pattern mining, arc classification)
Layer 3a: Inference   (causal, game-theoretic, motivation)
Layer 2: Hypergraph   (entities, situations, participation, state versioning, causal links)
Layer 1: Storage      (KV trait + RocksDB / in-memory)
Layer 0: Types        (UUIDs, serialization, Allen intervals)
```

The full reference — every TensaQL clause, every REST endpoint, every MCP tool, every KV prefix, every algorithm and citation — lives in [**`documentation/TENSA_REFERENCE.md`**](documentation/TENSA_REFERENCE.md). This README is the on-ramp.

---

## Installation

### Requirements

- **Rust 1.83+** with `cargo` ([rustup](https://rustup.rs/))
- **C++ toolchain** (for the default RocksDB backend) — MSVC on Windows, `clang` + `libclang-dev` on Linux, Xcode CLT on macOS
- **8 GB RAM** for development builds; the release binary itself runs in ~200 MB
- *(optional)* an LLM endpoint — local (Ollama, vLLM, llama.cpp) or hosted (Anthropic, OpenRouter, Gemini, Bedrock)

### Get the source

```bash
git clone https://github.com/arperon-labs/tensa.git
cd tensa
```

### Quick build matrix

```bash
# Pure-Rust, no RocksDB, no server — fastest, useful for CI / library embedding
cargo build --no-default-features

# Default: with RocksDB
cargo build

# Server (REST API)
cargo build --release --features server

# Server + MCP + embeddings + document parsing + web ingest + studio chat
cargo build --release --features "server,mcp,embedding,docparse,web-ingest,studio-chat"

# Adversarial wargaming layer (D12)
cargo build --release --features "server,adversarial"
```

### Feature flags

| Flag | What it adds |
|---|---|
| `rocksdb` *(default)* | Persistent KV store. Disable with `--no-default-features` for in-memory only. |
| `server` | REST API server (axum + tower). Required for `tensa-server` binary. |
| `mcp` | Model Context Protocol server (`tensa-mcp` binary). |
| `cli` | `tensa-cli` binary. |
| `embedding` | ONNX semantic embeddings via `ort` + `tokenizers`. |
| `docparse` | PDF (`lopdf`) + DOCX (`docx-rs`) ingestion. |
| `web-ingest` | URL extraction + RSS/Atom feed ingestion. |
| `gemini`, `bedrock` | Additional LLM providers. |
| `disinfo` *(default)* | Dual fingerprints, CIB detection, claim tracking, archetypes. |
| `generation` | Narrative generation (commitments, fabula/sjužet, dramatic irony, scene-sequel, …). |
| `adversarial` | SUQR rationality, wargame loop, DISARM TTPs, counter-narrative generation. |
| `studio-chat` | `/studio/chat` SSE endpoint for the Studio agent. |

### Run the tests

```bash
cargo test --no-default-features                # core (~600 tests, fastest)
cargo test                                      # default features (~1500 tests)
cargo test --features "server,mcp,embedding"    # full surface
```

---

## Running the server

### Bare minimum

```bash
cargo run --release --features server
# Listens on 0.0.0.0:3000
```

```bash
curl http://localhost:3000/health
# {"status":"ok"}
```

### With an LLM provider

LLM resolution priority: `LOCAL_LLM_URL` > `OPENROUTER_API_KEY` > `ANTHROPIC_API_KEY`.

```bash
# Local (Ollama, vLLM, LiteLLM, llama.cpp — anything OpenAI-compatible)
LOCAL_LLM_URL=http://localhost:11434 \
TENSA_MODEL=qwen3:32b \
cargo run --release --features server

# OpenRouter
OPENROUTER_API_KEY=sk-or-... \
TENSA_MODEL=anthropic/claude-sonnet-4 \
cargo run --release --features server

# Anthropic direct
ANTHROPIC_API_KEY=sk-ant-... \
TENSA_MODEL=claude-sonnet-4-20250514 \
cargo run --release --features server
```

You can also hot-swap providers at runtime: `PUT /settings/llm` with a JSON body.

### Common environment variables

| Variable | Purpose | Default |
|---|---|---|
| `TENSA_ADDR` | Bind address | `0.0.0.0:3000` |
| `TENSA_DATA_DIR` | RocksDB data directory | `./tensa_server_data` |
| `TENSA_MODEL` | LLM model name | provider default |
| `TENSA_EMBEDDING_MODEL` | Path to ONNX model dir (must contain `model.onnx` + `tokenizer.json`) | hash embedder |
| `RUST_LOG` | Log filter, e.g. `tensa=debug` | `info` |

---

## Using the MCP server

TENSA ships **178 MCP tools** covering every CRUD operation, query, inference job, narrative workshop primitive, fuzzy operator, and synthetic-generation pipeline. Plug it into any MCP-aware client (Claude Desktop, Claude Code, Cursor, etc).

### Build the MCP binary

```bash
cargo build --release --features mcp
# → target/release/tensa-mcp
```

The MCP binary speaks **stdio**. It can run **embedded** (direct library access — no HTTP) or **proxy** to a running `tensa-server`.

### Wire it into Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tensa": {
      "command": "/absolute/path/to/target/release/tensa-mcp",
      "args": ["--data-dir", "/absolute/path/to/tensa_server_data"]
    }
  }
}
```

Restart Claude Desktop. You should see **178 tools** appear (look for `mcp__tensa__*` in the tool picker).

### Wire it into Claude Code

```bash
claude mcp add tensa /absolute/path/to/target/release/tensa-mcp \
  --args "--data-dir,/absolute/path/to/tensa_server_data"
```

### Common MCP tools

| Tool | What it does |
|---|---|
| `query` | Run a TensaQL query. The Swiss Army knife. |
| `infer` | Submit causal / game-theoretic / motivation inference jobs. |
| `ingest_text` | LLM-extract entities + situations from raw text. |
| `ingest_url`, `ingest_rss` | Pull from a URL or feed. |
| `create_narrative`, `list_narratives` | Manage corpora. |
| `create_entity`, `create_situation`, `add_participant` | Manual CRUD. |
| `ask` | RAG question over a narrative (multi-mode retrieval). |
| `run_workshop` | Three-tier critique on a narrative. |
| `check_continuity` | Validate prose against pinned facts. |
| `calibrate_surrogate`, `generate_synthetic_narrative` | EATH synthetic generation. |
| `simulate_opinion_dynamics` | Bounded-confidence opinion dynamics on hypergraphs. |
| `argumentation_gradual` | h-Categoriser / Max-Based / Card-Based ranking-based argumentation. |
| `fuzzy_aggregate`, `fuzzy_learn_measure` | Fuzzy aggregation + Choquet measure learning. |

Full catalog: [`documentation/TENSA_REFERENCE.md`](documentation/TENSA_REFERENCE.md) § MCP Tools.

---

## Skills (chat-side tool packs)

Skills are **markdown bundles** that teach an LLM agent how to use TENSA's MCP tools idiomatically. They are baked into the binary at compile time via `include_str!` — the compiled tool set ships with them.

### Bundled skills

| Skill | What it teaches |
|---|---|
| `tensa` | Translating natural-language questions into TensaQL queries. |
| `tensa-writer` | Novel-writing workflow: pinned facts, workshop, revisions, continuity. |
| `tensa-synth` | EATH calibration → generation → fidelity → significance pipeline. |
| `tensa-fuzzy` | Fuzzy logic operators (t-norms, OWA, Choquet, fuzzy Allen, Mamdani). |
| `tensa-graded` | Gradual argumentation + ranking-supervised measure learning. |
| `tensa-opinion-dynamics` | Bounded-confidence opinion dynamics, phase transitions, echo-chamber detection. |
| `tensa-reconstruction` | SINDy hypergraph reconstruction from dynamics. |
| `studio-ui` | Studio UI navigation reference (used by the Studio chat agent). |

The skill source lives in [`skills/`](skills/). They are auto-registered by the `studio-chat` feature; clients enumerate them via `GET /studio/chat/skills`.

### Installing a skill into Claude Code

If you want Claude Code itself to follow one of these skills (rather than the in-Studio agent), copy the bundle into your Claude Code skills directory:

```bash
# Linux / macOS
mkdir -p ~/.claude/skills
cp skills/tensa.md ~/.claude/skills/

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\skills" | Out-Null
Copy-Item skills\tensa.md "$env:USERPROFILE\.claude\skills\"
```

Restart Claude Code. The skill becomes available via `/<skill-name>`.

### Writing your own skill

A skill is just a markdown file. The first paragraph after the YAML frontmatter tells the agent *when to use* the skill; the rest is reference material the agent reads when invoked. See `skills/tensa.md` for a worked example.

---

## TensaQL in 60 seconds

```sql
-- Find high-confidence actors named in a specific narrative
MATCH (e:Actor)
WHERE e.confidence > 0.8 AND e.narrative_id = "case-alpha"
RETURN e
LIMIT 10

-- Allen-temporal: situations that happened BEFORE a given timestamp
MATCH (s:Situation) AT s.temporal BEFORE "2025-01-01T00:00:00Z" RETURN s

-- Vector + spatial + temporal in one query
MATCH (s:Situation)
NEAR(s, "violent confrontation", 5)
SPATIAL s.spatial WITHIN 10.0 KM OF (40.7128, -74.0060)
RETURN s

-- Submit a causal inference job (returns a job id; poll /jobs/{id}/result)
INFER CENTRALITY("case-alpha")

-- RAG question over a narrative with multi-mode retrieval
ASK "Who killed the victim and why?"
OVER "case-alpha"
MODE drift
RESPOND AS "bullet points"
SUGGEST

-- Cross-narrative discovery
DISCOVER PATTERNS ACROSS NARRATIVES WHERE genre = "noir"

-- Calibrate a surrogate model + generate synthetic data
CALIBRATE SURROGATE USING 'eath' FOR "case-alpha"
GENERATE NARRATIVE "case-alpha-synth" LIKE "case-alpha"
  USING SURROGATE 'eath' SEED 42 STEPS 100

-- Fuzzy logic with custom t-norm + aggregator
MATCH (e:Actor) WHERE e.confidence > 0.6 RETURN e
WITH TNORM 'lukasiewicz' AGGREGATE OWA
```

Full grammar: [`documentation/TENSA_REFERENCE.md`](documentation/TENSA_REFERENCE.md) § TensaQL.

---

## REST API quick examples

```bash
# Create an entity
curl -X POST http://localhost:3000/entities \
  -H 'Content-Type: application/json' \
  -d '{
    "entity_type": "Actor",
    "properties": {"name": "Raskolnikov", "age": 23},
    "narrative_id": "crime-and-punishment",
    "confidence": 0.9
  }'

# Run a TensaQL query
curl -X POST http://localhost:3000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "MATCH (e:Actor) WHERE e.confidence > 0.8 RETURN e LIMIT 5"}'

# Ingest raw text (LLM extracts entities + situations)
curl -X POST http://localhost:3000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text": "...", "narrative_id": "case-alpha", "enrich": true}'

# Ask a RAG question
curl -X POST http://localhost:3000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Who is the protagonist?", "narrative_id": "case-alpha", "mode": "hybrid"}'
```

Full endpoint reference: [`documentation/TENSA_REFERENCE.md`](documentation/TENSA_REFERENCE.md) § REST API.

---

## Studio UI

Studio is a React + TypeScript console for analysts and writers — the human-facing surface for the engine. It lives in a **separate repo** and is distributed as a Docker image rather than as source. If you only need the engine, skip this section.

To run TENSA *headless* (engine + REST + MCP), nothing in this repo requires Studio.

---

## Documentation map

- [`documentation/TENSA_REFERENCE.md`](documentation/TENSA_REFERENCE.md) — **canonical reference**. Every clause, endpoint, tool, KV prefix, algorithm, and citation. Start here for anything beyond the on-ramp.
- [`license/`](license/) — dual-licensing materials (AGPL-3.0 + commercial).
- [`skills/`](skills/) — bundled skill markdown.
- [`src/`](src/) — engine source. Module layout mirrors the architecture diagram above.
- [`tests/`](tests/) — unit + integration + benchmark suites.

---

## Contributing

We accept contributions under the [Contributor License Agreement](license/CLA.md). See [`license/CONTRIBUTING.md`](license/CONTRIBUTING.md) for the workflow.

Before you open a PR:

1. `cargo fmt`
2. `cargo clippy -- -D warnings`
3. `cargo test --no-default-features`
4. Add or update tests next to the change.
5. If you alter behavior, an API, or a file format, update [`documentation/TENSA_REFERENCE.md`](documentation/TENSA_REFERENCE.md) in the same PR.

---

## License

TENSA is **dual-licensed**. Choose the license that fits your use case:

- **[GNU AGPL-3.0](license/LICENSE)** — free, for research, open-source projects, and internal use. Requires that you release the source of any product that embeds or network-serves TENSA under AGPL-3.0 too.
- **Commercial license** from Arperon s.r.o. — for proprietary products, SaaS, and any use that cannot accept AGPL-3.0 obligations. See [`license/COMMERCIAL-LICENSE-TERMS.md`](license/COMMERCIAL-LICENSE-TERMS.md) or email **licensing@arperon.com**.

See [`license/DUAL-LICENSING.md`](license/DUAL-LICENSING.md) for how to choose and a plain-English FAQ.

Copyright © 2026 Arperon s.r.o. TENSA is a trademark of Arperon s.r.o.
