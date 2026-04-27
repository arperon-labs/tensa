# Bundle deployment — TENSA + Studio in one compose

The fastest way to evaluate TENSA. Pulls **pre-built images** from GitHub Container Registry — no source checkout, no compile, no Rust toolchain. Two services on a private bridge network: `tensa-core` (engine, REST, MCP, RocksDB) and `tensa-studio` (web UI, nginx, talks to core internally).

```
   browser :8080
       │
       │ HTTP, same-origin
       ▼
   ┌─────────────────────────┐        ┌─────────────────────────┐
   │  tensa-studio (nginx)   │  /api  │  tensa-core             │
   │  ghcr.io/arperon-labs/  │ ─────→ │  ghcr.io/arperon-labs/  │
   │  tensa-studio:latest    │        │  tensa:latest           │
   │  port 8080              │        │  port 3000 (internal)   │
   └─────────────────────────┘        └─────────────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │  named volume    │
                                        │  tensa-data      │
                                        │  RocksDB at /data│
                                        └──────────────────┘
```

## Quick start

```bash
# 1. From the repo root, copy the env template
cp docker-bundle/.env.example docker-bundle/.env
# 2. Edit it — at minimum set ONE of LOCAL_LLM_URL / OPENROUTER_API_KEY / ANTHROPIC_API_KEY
$EDITOR docker-bundle/.env
# 3. Launch
docker compose -f docker-bundle/docker-compose.yml --env-file docker-bundle/.env up -d
# 4. Open Studio
open http://localhost:8080      # macOS
xdg-open http://localhost:8080  # Linux
start http://localhost:8080     # Windows
```

The first `up` pulls roughly 100-200 MB of layers and starts up in ~10-20 seconds. Subsequent runs use the cached images and start in seconds.

## Lifecycle

```bash
# Status
docker compose -f docker-bundle/docker-compose.yml ps

# Logs
docker compose -f docker-bundle/docker-compose.yml logs -f
docker compose -f docker-bundle/docker-compose.yml logs -f tensa-core
docker compose -f docker-bundle/docker-compose.yml logs -f tensa-studio

# Stop / restart (state preserved)
docker compose -f docker-bundle/docker-compose.yml stop
docker compose -f docker-bundle/docker-compose.yml restart

# Tear down (containers gone, volumes preserved)
docker compose -f docker-bundle/docker-compose.yml down

# DESTRUCTIVE: containers + named volumes (your RocksDB data is gone)
docker compose -f docker-bundle/docker-compose.yml down -v
```

## Updating

```bash
docker compose -f docker-bundle/docker-compose.yml pull       # pull newer images
docker compose -f docker-bundle/docker-compose.yml up -d      # recreate with the new ones
```

The `tensa-data` named volume survives image upgrades — your KV store is preserved.

## Versioning

Both images use semver-aligned tags. Pin to a specific minor for production:

```yaml
services:
  tensa-core:
    image: ghcr.io/arperon-labs/tensa:0.79.2
  tensa-studio:
    image: ghcr.io/arperon-labs/tensa-studio:0.79.2
```

**Compatibility rule:** matching minor versions are guaranteed compatible. Patch-level skew within a minor is fine. Cross-minor compatibility is documented per release.

## Configuration

Everything is set via environment variables in `docker-bundle/.env` (or in your shell). See [`.env.example`](.env.example) for the full list. The two essentials:

- **One LLM provider.** TENSA reads `LOCAL_LLM_URL`, `OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY` (in that priority order). Without one, ingestion / RAG / inference jobs that require an LLM will return errors; CRUD and TensaQL queries still work.
- **Optional embedding model.** Drop a directory containing `model.onnx` + `tokenizer.json` somewhere on the host, mount it into the `tensa-core` service at `/models/<name>`, and set `TENSA_EMBEDDING_MODEL=/models/<name>` in `.env`. Without it, TENSA falls back to a hash embedder.

## Changing the Studio port

```bash
STUDIO_HOST_PORT=9090 docker compose -f docker-bundle/docker-compose.yml up -d
# or set STUDIO_HOST_PORT in .env
```

The internal port (8080) stays the same; only the host mapping changes.

## What if I want to build core from source?

This bundle pulls a published image. If you want to build core yourself (for example to add custom features), see [`../docker/README.md`](../docker/README.md) — the engine source is open and the Dockerfile is reproducible from a clean clone. Then point this compose at your local image:

```yaml
services:
  tensa-core:
    image: tensa:my-build       # your locally built tag
    # ...everything else identical
```

## What about Studio?

Studio is **distributed as a Docker image only**. The source is not currently public, so there is no Dockerfile in this repo for it — you pull the published image. For a production deployment, treat the Studio image as a sealed artifact: pin a version, run it, log issues against the public repo with reproduction steps + container logs.

## Troubleshooting

**Studio shows 502 Bad Gateway when calling the API** — `tensa-core` isn't healthy yet. The Studio service has `depends_on: tensa-core { condition: service_healthy }` but if you started Studio first (or restarted it without core), give core 10-15 seconds. Check `docker compose ps` and the core logs.

**`docker compose pull` fails with "denied"** — the images may not be public yet, or you're hitting GHCR rate limits unauthenticated. `docker login ghcr.io -u <username>` with a PAT and try again.

**Container starts but `curl http://localhost:8080` connects to nothing** — Studio is up but you mapped the port differently. Check `docker compose ps` for the actual host:container mapping.

**LLM calls fail with "no provider configured"** — your `.env` is missing all three of `LOCAL_LLM_URL`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`. Set at least one and `docker compose restart tensa-core`.

**On Linux, host LLM (e.g. Ollama at `localhost:11434`) is unreachable** — `host.docker.internal` only works on Docker Desktop (macOS/Windows). On Linux, use the host's bridge IP (`172.17.0.1` on default Docker installs) or run your local LLM in a sibling container on the same bundle network.
