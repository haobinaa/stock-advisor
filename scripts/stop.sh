#!/usr/bin/env bash
# 停止 Stock Advisor
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if pid=$(get_pid); then
    echo "正在停止 (PID: $pid)..."
    kill "$pid" 2>/dev/null
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "已停止"
else
    echo "未在运行"
    # 清理端口残留
    lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
fi
