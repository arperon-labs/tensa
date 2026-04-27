# Docker — TENSA container build

Multi-stage Dockerfile that produces a small, self-contained runtime image of the TENSA REST API server.

## What's in `docker/`

| File | Purpose |
|---|---|
| [`Dockerfile`](Dockerfile) | Two-stage build: `rust:slim-bookworm` builder → `debian:bookworm-slim` runtime. Final image ~90-130 MB depending on feature set. |
| [`docker-compose.yml`](docker-compose.yml) | Convenience wrapper for local dev: builds, runs, mounts data + model volumes, wires env vars. |
| [`build.sh`](build.sh) / [`build.ps1`](build.ps1) | Cross-platform wrappers around `docker build` with sensible defaults. |
| [`.dockerignore`](../.dockerignore) | Lives at the repo root (Docker requirement). Excludes `target/`, `.git/`, papers, corpora, models, studio source, etc. from the build context. |

## Quick start

### Build

From the **repo root**:

```bash
# Linux / macOS
./docker/build.sh

# Windows
.\docker\build.ps1
```

Or directly:

```bash
docker build -f docker/Dockerfile -t tensa:latest .
```

### Run

```bash
# Smallest possible run — no LLM, no embedding, just the engine
docker run --rm -p 3000:3000 -v tensa-data:/data tensa:latest

# Health check
curl http://localhost:3000/health
# {"status":"ok"}
```

### Run with an LLM provider

LLM resolution priority: `LOCAL_LLM_URL` > `OPENROUTER_API_KEY` > `ANTHROPIC_API_KEY`.

```bash
# OpenRouter
docker run -d --name tensa \
  -p 3000:3000 \
  -v tensa-data:/data \
  -e OPENROUTER_API_KEY=sk-or-... \
  -e TENSA_MODEL=anthropic/claude-sonnet-4 \
  tensa:latest

# Local LLM (Ollama, vLLM, etc) — note host.docker.internal on macOS/Windows
docker run -d --name tensa \
  -p 3000:3000 \
  -v tensa-data:/data \
  -e LOCAL_LLM_URL=http://host.docker.internal:11434 \
  -e TENSA_MODEL=qwen3:32b \
  tensa:latest
```

### Run with docker compose

The compose file picks up env vars from a `.env` next to it (or from the shell):

```bash
# .env (next to docker-compose.yml or your shell)
OPENROUTER_API_KEY=sk-or-...
TENSA_MODEL=anthropic/claude-sonnet-4
RUST_LOG=tensa=info
```

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs -f tensa
docker compose -f docker/docker-compose.yml down
```

## Changing the port

The bind address is governed by `TENSA_ADDR`. With Docker there are two ways to expose the API on a port other than 3000.

### Recommended: remap the host port, keep the container internal at 3000

```bash
docker run -p 8080:3000 tensa:latest
```

The container still listens on 3000; Docker maps host `:8080` → container `:3000`. The image's healthcheck stays valid (it queries `127.0.0.1:3000` internally). **No rebuild required.**

For compose:
```yaml
services:
  tensa:
    ports:
      - "8080:3000"
```

### Fully relocate (also moves the internal port)

```bash
docker run -p 8080:8080 -e TENSA_ADDR=0.0.0.0:8080 tensa:latest
```

> ⚠️ **Healthcheck caveat.** The image's baked-in healthcheck hits `http://127.0.0.1:3000/health`. If you change the internal port via `-e TENSA_ADDR=...`, the healthcheck will fail and Docker will mark the container `unhealthy` — even though it's serving correctly. Two ways to handle it:
>
> 1. **Just remap the host port** (option above) — by far the simplest.
> 2. Rebuild the image with `--build-arg` for the new port and edit `HEALTHCHECK CMD` in the Dockerfile, or override at runtime with `--health-cmd "curl -fsS http://127.0.0.1:8080/health"`.

The `EXPOSE 3000` line in the Dockerfile is purely informational — `-p` flags do the actual binding, so it doesn't need to match.

## Customising the build

### Custom feature set

The full default set is roughly the same as the `scripts/tensa.{sh,ps1}` launcher uses. To trim the image:

```bash
# Engine + REST + MCP only — smallest useful image, ~70 MB
./docker/build.sh -t tensa:slim -f "server,mcp,rocksdb"

# Add embeddings (pulls in ONNX runtime — adds ~80 MB)
./docker/build.sh -t tensa:embed -f "server,mcp,rocksdb,embedding"
```

### ONNX embedding model

The image does **not** bake in ONNX models. Mount a host directory and point the env var at it:

```bash
docker run -d \
  -p 3000:3000 \
  -v tensa-data:/data \
  -v /path/to/models:/models:ro \
  -e TENSA_EMBEDDING_MODEL=/models/all-MiniLM-L6-v2 \
  tensa:embed
```

The directory must contain `model.onnx` and `tokenizer.json`.

## Publishing to a registry

### GitHub Container Registry (`ghcr.io/arperon-labs/tensa`)

```bash
# One-time: log in (use a GitHub PAT with packages:write scope)
echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin

# Build + tag + push
./docker/build.sh -t ghcr.io/arperon-labs/tensa:v0.79.2 --push
docker tag ghcr.io/arperon-labs/tensa:v0.79.2 ghcr.io/arperon-labs/tensa:latest
docker push ghcr.io/arperon-labs/tensa:latest
```

After the first push, set the package visibility to public (or keep it private for a paid distribution channel) under github.com/orgs/arperon-labs/packages.

### Docker Hub

```bash
docker login
./docker/build.sh -t arperon/tensa:v0.79.2 --push
```

## Image layout

```
/app/tensa-server          # the only binary; ENTRYPOINT
/data/                     # persistent KV store (RocksDB)        — VOLUME
/models/                   # optional ONNX embedding models       — VOLUME (read-only)
```

- **User:** non-root UID 10001 (group 0 for OpenShift compatibility).
- **Port:** 3000 (set `TENSA_ADDR=0.0.0.0:NNNN` to change).
- **Healthcheck:** `curl /health` every 30s.

## What is **not** in the image

- **Studio UI** — distributed separately; this is the headless engine.
- **Source code** — only the stripped release binary ships.
- **LLM API keys** — supplied at runtime via env vars or compose secrets.
- **ONNX models** — supplied at runtime via volume mount.
- **Tests, benchmarks, papers** — excluded by `.dockerignore`.

## Troubleshooting

**`error: linker 'cc' not found`** during build — the builder stage needs `clang` + `build-essential`; the Dockerfile installs them. If you see this, you're probably running the build outside the Dockerfile.

**RocksDB compile takes forever** — it does. Use `--mount=type=cache,target=/build/target` (already wired in the Dockerfile) to cache between builds.

**Image is huge (~500 MB+)** — you accidentally built with `--no-default-features` removed and pulled in `embedding` + every test crate. Use `-f "server,mcp,rocksdb"` for the minimum useful image.

**`reqwest: TLS error`** — base image is missing `ca-certificates`. The Dockerfile installs them; if you're using a different base, add `apt-get install -y ca-certificates`.
