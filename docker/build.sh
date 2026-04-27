#!/usr/bin/env bash
# docker/build.sh — Build the TENSA container image from the repo root.
#
# Usage:
#   ./docker/build.sh                           # default: tensa:latest, full features
#   ./docker/build.sh -t myreg/tensa:v0.79.2    # custom tag
#   ./docker/build.sh -f "server,mcp"           # custom feature list (quoted)
#   ./docker/build.sh --push                    # build then push to registry

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TAG="tensa:latest"
FEATURES="server,studio-chat,inference,web-ingest,docparse,generation,adversarial,gemini,bedrock,mcp,rocksdb,disinfo"
PUSH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)       TAG="$2"; shift 2 ;;
        -f|--features)  FEATURES="$2"; shift 2 ;;
        --push)         PUSH=1; shift ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0
            ;;
        *)              echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "  Building $TAG"
echo "  Features: $FEATURES"
echo "  Context:  $ROOT_DIR"
echo ""

cd "$ROOT_DIR"

DOCKER_BUILDKIT=1 docker build \
    --file docker/Dockerfile \
    --build-arg "TENSA_FEATURES=$FEATURES" \
    --tag "$TAG" \
    .

echo ""
echo "  Built: $TAG"

if [[ $PUSH -eq 1 ]]; then
    echo ""
    echo "  Pushing $TAG..."
    docker push "$TAG"
fi
