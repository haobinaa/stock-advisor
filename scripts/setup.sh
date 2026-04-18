#!/usr/bin/env bash
# 环境检测 + venv 创建 + 依赖安装
# 用法: ./scripts/setup.sh
#   被 start.sh 自动调用，也可单独运行

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

# ── 检测 Python ──────────────────────────────────────────────

find_python() {
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            if [ -n "$ver" ] && python3 -c "exit(0 if tuple(map(int,'$ver'.split('.'))) >= tuple(map(int,'$PYTHON_MIN_VERSION'.split('.'))) else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

if [ -f "$VENV_PYTHON" ]; then
    echo "[setup] venv 已存在: $VENV_DIR"
    PYTHON_CMD="$VENV_PYTHON"
else
    echo "[setup] 检测 Python 版本 (>= $PYTHON_MIN_VERSION)..."
    PYTHON_CMD=$(find_python) || {
        echo "[setup] 错误: 未找到 Python >= $PYTHON_MIN_VERSION"
        echo "  请安装 Python 3.10+ 后重试"
        exit 1
    }
    PY_VER=$("$PYTHON_CMD" --version 2>&1)
    echo "[setup] 找到: $PYTHON_CMD ($PY_VER)"

    # ── 创建 venv ────────────────────────────────────────────
    echo "[setup] 创建虚拟环境: $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "[setup] venv 创建完成"
fi

# ── 安装/更新依赖 ────────────────────────────────────────────

if [ -f "$REQUIREMENTS" ]; then
    echo "[setup] 安装依赖..."
    "$VENV_PIP" install -q -r "$REQUIREMENTS"
    echo "[setup] 依赖安装完成"
else
    echo "[setup] 警告: 未找到 requirements.txt"
fi

# ── 创建数据目录 ─────────────────────────────────────────────

mkdir -p "$DATA_DIR"

echo "[setup] 环境就绪"
