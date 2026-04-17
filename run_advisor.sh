#!/usr/bin/env bash
# Stock Advisor - 启动/停止/状态管理脚本
# 用法:
#   ./run_advisor.sh start   # 启动服务
#   ./run_advisor.sh stop    # 停止服务
#   ./run_advisor.sh restart # 重启服务
#   ./run_advisor.sh status  # 查看状态
#   ./run_advisor.sh log     # 查看日志

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PID_FILE="$PROJECT_DIR/app/data/advisor.pid"
LOG_FILE="$PROJECT_DIR/app/data/advisor.log"
PORT=5001

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: .venv not found. Run: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

get_pid() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    return 1
}

do_start() {
    if pid=$(get_pid); then
        echo "Already running (PID: $pid)"
        echo "Visit: http://localhost:$PORT"
        return 0
    fi

    # Kill anything on the port
    lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1

    echo "Starting Stock Advisor..."
    mkdir -p "$(dirname "$PID_FILE")"

    cd "$PROJECT_DIR"
    nohup "$VENV_PYTHON" -m app.app > "$LOG_FILE" 2>&1 &
    local new_pid=$!

    # Detach from shell: disown so SIGHUP won't kill it
    disown "$new_pid" 2>/dev/null || true

    echo "$new_pid" > "$PID_FILE"
    sleep 2

    if kill -0 "$new_pid" 2>/dev/null; then
        echo "Started (PID: $new_pid)"
        echo "Visit: http://localhost:$PORT"
        echo "Log:   $LOG_FILE"
    else
        echo "Failed to start. Check log:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

do_stop() {
    if pid=$(get_pid); then
        echo "Stopping (PID: $pid)..."
        kill "$pid" 2>/dev/null
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "Stopped"
    else
        echo "Not running"
        # Clean up any orphan on the port
        lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    fi
}

do_status() {
    if pid=$(get_pid); then
        echo "Running (PID: $pid)"
        echo "URL: http://localhost:$PORT"
        curl -s -o /dev/null -w "Health: HTTP %{http_code} (%{time_total}s)\n" "http://localhost:$PORT/watchlist" 2>/dev/null || echo "Health: not responding"
    else
        echo "Not running"
    fi
}

do_log() {
    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE"
    else
        echo "No log file"
    fi
}

case "${1:-status}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 1; do_start ;;
    status)  do_status ;;
    log)     do_log ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
