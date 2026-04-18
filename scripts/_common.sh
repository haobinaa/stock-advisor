#!/usr/bin/env bash
# 公共变量 — 被其他脚本 source 引用

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

DATA_DIR="$PROJECT_DIR/app/data"
PID_FILE="$DATA_DIR/advisor.pid"
LOG_FILE="$DATA_DIR/advisor.log"

PORT=5001
PYTHON_MIN_VERSION="3.10"

# 获取正在运行的 PID，未运行则返回 1
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
