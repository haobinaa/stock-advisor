# Stock Advisor

基于 [Kronos](https://huggingface.co/NeoQuasar/Kronos-small) 时序预测模型的 A 股智能分析系统。结合深度学习价格预测、技术指标分析、资金流向和融资融券数据，为个股提供多维度综合评分。

## 功能

- **个股分析** — Kronos 模型预测 + 技术面 + 资金面 + 融资融券，六维加权评分
- **自选股管理** — 添加/删除自选，批量分析并按评分排序
- **全市场扫描** — 扫描沪深 300 + 中证 500 成分股，筛选高分标的
- **回测验证** — 记录历史预测，自动回测计算准确率，持续评估模型效果
- **权重优化** — 基于回测数据自动优化六维评分权重
- **模型微调** — 支持在 A 股数据上微调 Kronos 模型

## 项目结构

```
stock-advisor/
├── app/                    # Flask Web 应用
│   ├── app.py              # 主入口 & API 路由
│   ├── config.py           # 配置项
│   ├── models/             # 数据模型
│   ├── services/           # 业务逻辑
│   │   ├── analyzer.py     # 综合分析 & 评分
│   │   ├── backtest.py     # 回测记录 & 验证
│   │   ├── data_fetcher.py # 行情数据获取 (akshare)
│   │   ├── predictor.py    # Kronos 模型预测
│   │   ├── scanner.py      # 全市场扫描
│   │   ├── technical.py    # 技术指标分析
│   │   ├── weight_optimizer.py  # 权重优化
│   │   ├── prediction_filter.py # 预测结果过滤
│   │   └── finetune_manager.py  # 微调管理
│   └── templates/          # HTML 页面
├── model/                  # Kronos 模型定义
├── finetune_csv/           # 微调训练脚本
├── requirements.txt
└── run_advisor.sh          # 启动/停止管理脚本
```

## 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.0+（GPU 推荐，支持 CUDA / MPS / CPU）

### 安装

```bash
git clone https://github.com/haobinaa/stock-advisor.git
cd stock-advisor
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 启动

```bash
# 使用管理脚本
./run_advisor.sh start

# 或直接运行
python -m app.app
```

访问 http://localhost:5001

### 管理命令

```bash
./run_advisor.sh start    # 启动服务
./run_advisor.sh stop     # 停止服务
./run_advisor.sh restart  # 重启服务
./run_advisor.sh status   # 查看状态
./run_advisor.sh log      # 查看日志
```

## 评分维度

| 维度 | 默认权重 | 数据来源 |
|------|---------|---------|
| 预期收益率 | 30% | Kronos 模型预测 |
| 最大回撤 | 20% | 预测路径分析 |
| 不确定性 | 15% | 多次采样标准差 |
| 技术面 | 15% | MACD / RSI / 布林带等 |
| 资金流向 | 10% | 主力净流入 |
| 融资融券 | 10% | 融资余额趋势 |

权重可通过回测数据自动优化。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `KRONOS_MODEL` | `NeoQuasar/Kronos-small` | Kronos 模型名称 |
| `KRONOS_TOKENIZER` | `NeoQuasar/Kronos-Tokenizer-base` | Tokenizer 名称 |
| `FLASK_DEBUG` | `0` | 调试模式 |

## License

MIT
