#!/usr/bin/env bash
# scripts/tensa.sh — Start/stop/status for the TENSA API server
# Usage: ./scripts/tensa.sh [status|start|stop|restart|build|rebuild|logs]
#
# Public/community-facing launcher. Manages a single service — the TENSA REST
# API server — and writes pidfiles + logs to .pids/ and .logs/ under the repo
# root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDDIR="$SCRIPT_DIR/.pids"
LOGDIR="$SCRIPT_DIR/.logs"
mkdir -p "$PIDDIR" "$LOGDIR"

SERVICE_NAME="api"
SERVICE_PORT="4350"
FEATURES="server,studio-chat,embedding,inference,web-ingest,docparse,generation,adversarial,gemini,bedrock,mcp"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

# --- Helpers ---

is_running() {
    local pidfile="$PIDDIR/$SERVICE_NAME.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$pidfile"
    fi
    return 1
}

get_pid() {
    cat "$PIDDIR/$SERVICE_NAME.pid" 2>/dev/null || echo ""
}

# --- Service control ---

start_api() {
    if is_running; then
        echo -e "  ${YELLOW}API already running${NC} (pid $(get_pid))"
        return 0
    fi
    echo -e "  ${CYAN}Starting TENSA API on :$SERVICE_PORT...${NC}"
    cd "$SCRIPT_DIR"
    TENSA_ADDR="0.0.0.0:$SERVICE_PORT" \
    cargo run --release --bin tensa-server --features "$FEATURES" \
        > "$LOGDIR/api.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PIDDIR/$SERVICE_NAME.pid"
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "  ${GREEN}API started${NC} (pid $pid) -> http://localhost:$SERVICE_PORT"
    else
        echo -e "  ${RED}API failed to start${NC} - check .logs/api.log"
        rm -f "$PIDDIR/$SERVICE_NAME.pid"
        return 1
    fi
}

stop_api() {
    if is_running; then
        local pid
        pid=$(get_pid)
        echo -e "  ${CYAN}Stopping API${NC} (pid $pid)..."
        if command -v taskkill &>/dev/null; then
            taskkill //F //T //PID "$pid" >/dev/null 2>&1 || kill "$pid" 2>/dev/null || true
        else
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDDIR/$SERVICE_NAME.pid"
        echo -e "  ${GREEN}API stopped${NC}"
    else
        echo -e "  ${YELLOW}API not running${NC}"
    fi
}

show_status() {
    echo ""
    echo -e "${BOLD}  TENSA API${NC}"
    echo -e "  ─────────────────────────────────────"
    if is_running; then
        echo -e "  ${GREEN}● running${NC}  pid=$(get_pid)  http://localhost:$SERVICE_PORT"
        echo ""
        echo -e "  ${CYAN}Logs:${NC} .logs/api.log"
        echo -e "  ${CYAN}Tail:${NC} tail -f .logs/api.log"
    else
        echo -e "  ${RED}○ stopped${NC}"
    fi
    echo ""
    echo -e "${BOLD}  Features${NC}"
    echo -e "  ─────────────────────────────────────"
    echo -e "  $FEATURES"
    echo ""
}

# --- Build ---

build_api() {
    echo ""
    echo -e "  ${CYAN}Building TENSA (release, full feature set)...${NC}"
    cd "$SCRIPT_DIR"
    if cargo build --release --bin tensa-server --features "$FEATURES"; then
        echo -e "  ${GREEN}Build succeeded${NC}"
        return 0
    else
        echo -e "  ${RED}Build failed${NC}"
        return 1
    fi
}

# --- Main ---

CMD="${1:-status}"

case "$CMD" in
    status|s)
        show_status
        ;;
    start)
        echo ""
        start_api
        echo ""
        show_status
        ;;
    stop)
        echo ""
        stop_api
        echo ""
        show_status
        ;;
    restart)
        echo ""
        stop_api
        start_api
        echo ""
        show_status
        ;;
    build)
        build_api
        echo ""
        ;;
    rebuild)
        echo ""
        stop_api
        if build_api; then start_api; fi
        echo ""
        show_status
        ;;
    logs)
        tail -f "$LOGDIR/api.log"
        ;;
    *)
        echo ""
        echo -e "${BOLD}Usage:${NC} ./scripts/tensa.sh [command]"
        echo ""
        echo -e "${BOLD}Commands:${NC}"
        echo "  status   Show service status (default)"
        echo "  start    Build (if needed) and start the API"
        echo "  stop     Stop the API"
        echo "  restart  Stop + start"
        echo "  build    cargo build --release with the full feature set"
        echo "  rebuild  Stop + build + start (after code changes)"
        echo "  logs     Tail the API log"
        echo ""
        ;;
esac
