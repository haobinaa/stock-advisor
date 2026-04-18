#!/usr/bin/env bash
# 查看 Stock Advisor 日志
# 用法:
#   ./scripts/logs.sh        # 最近 50 行
#   ./scripts/logs.sh -f     # 实时跟踪
#   ./scripts/logs.sh 100    # 最近 100 行

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if [ ! -f "$LOG_FILE" ]; then
    echo "日志文件不存在: $LOG_FILE"
    exit 1
fi

if [ "$1" = "-f" ]; then
    echo "实时跟踪日志 (Ctrl+C 退出)..."
    echo "---"
    tail -f "$LOG_FILE"
elif [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null; then
    tail -"$1" "$LOG_FILE"
else
    tail -50 "$LOG_FILE"
fi
