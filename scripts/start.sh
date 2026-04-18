#!/usr/bin/env bash
# 启动 Stock Advisor（自动检测环境）
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

# 环境检测 + 安装
bash "$SCRIPT_DIR/setup.sh"

# 检查是否已运行
if pid=$(get_pid); then
    echo "已在运行 (PID: $pid)"
    echo "访问: http://localhost:$PORT"
    exit 0
fi

# 清理端口残留
lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

echo "启动 Stock Advisor..."
mkdir -p "$DATA_DIR"

cd "$PROJECT_DIR"
nohup "$VENV_PYTHON" -m app.app > "$LOG_FILE" 2>&1 &
new_pid=$!
disown "$new_pid" 2>/dev/null || true
echo "$new_pid" > "$PID_FILE"

sleep 2

if kill -0 "$new_pid" 2>/dev/null; then
    echo "启动成功 (PID: $new_pid)"
    echo "访问: http://localhost:$PORT"
    echo "日志: $LOG_FILE"
else
    echo "启动失败，查看日志:"
    tail -20 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
