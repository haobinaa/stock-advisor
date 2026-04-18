#!/usr/bin/env bash
# 查看 Stock Advisor 运行状态

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if pid=$(get_pid); then
    echo "运行中 (PID: $pid)"
    echo "地址: http://localhost:$PORT"
    echo ""
    # 健康检查
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/watchlist" --max-time 5 2>/dev/null || echo "000")
    if [ "$http_code" = "200" ]; then
        echo "健康检查: OK (HTTP $http_code)"
    else
        echo "健康检查: 异常 (HTTP $http_code)"
    fi
    echo ""
    # 进程资源
    ps -p "$pid" -o pid,rss,vsz,%cpu,etime 2>/dev/null | head -2
else
    echo "未在运行"
fi
