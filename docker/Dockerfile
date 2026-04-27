# syntax=docker/dockerfile:1.7
#
# TENSA — multi-stage container build
#
# Stage 1: build the Rust release binary (with full or custom feature set).
# Stage 2: minimal runtime image — only the binary + a writable data dir.
#
# Build context is the repository root. Invoke from the repo root:
#
#   docker build -f docker/Dockerfile -t tensa:latest .
#
# Tunable via --build-arg:
#   TENSA_FEATURES   — cargo feature list (default: full set)
#   RUST_VERSION     — pinned Rust toolchain (default: 1.83)
#
# Resulting image (release, full features) is approximately 90-130 MB.

ARG RUST_VERSION=1.83

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-slim-bookworm AS builder

ARG TENSA_FEATURES="server,studio-chat,inference,web-ingest,docparse,generation,adversarial,gemini,bedrock,mcp,rocksdb,disinfo"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        clang \
        libclang-dev \
        pkg-config \
        cmake \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache Cargo registry + dependencies. We copy the manifests first, build a
# stub, and then copy real sources — so dependency compilation is reused on
# source-only changes.
COPY Cargo.toml Cargo.lock build.rs ./

# Stub source tree to prime the dependency cache.
RUN mkdir -p src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    echo 'fn main() {}' > benches/benchmarks.rs

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin tensa-server --features "${TENSA_FEATURES}" || true

# Now copy the actual source.
COPY src ./src
COPY tests ./tests
COPY benches ./benches
COPY skills ./skills
COPY documentation ./documentation
COPY license ./license

# Real build. The cache mounts mean only changed crates recompile.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --bin tensa-server --features "${TENSA_FEATURES}" \
 && cp target/release/tensa-server /tensa-server \
 && strip /tensa-server || true

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
# debian:bookworm-slim is small (~75 MB), has glibc + the C runtime that
# RocksDB needs, and lets us drop in `curl` for the healthcheck without
# fighting distroless. Switch to gcr.io/distroless/cc-debian12 if the
# healthcheck doesn't matter to you and you want the smallest possible image.
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --system --uid 10001 --gid 0 --create-home --home-dir /home/tensa tensa

WORKDIR /app
COPY --from=builder --chown=10001:0 /tensa-server /app/tensa-server

# Persistent data directory (volume mount target).
RUN mkdir -p /data && chown -R 10001:0 /data
ENV TENSA_DATA_DIR=/data
ENV TENSA_ADDR=0.0.0.0:3000

# Optional embedding model directory. Mount a host folder containing
# {model.onnx, tokenizer.json} and set TENSA_EMBEDDING_MODEL=/models/<name>.
RUN mkdir -p /models && chown -R 10001:0 /models

USER 10001:0
EXPOSE 3000
VOLUME ["/data", "/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:3000/health || exit 1

ENTRYPOINT ["/app/tensa-server"]
