# Kronos Stock Advisor — 设计文档

## 概述

基于 Kronos 金融 K 线基础模型构建的**股票分析与选股系统**。提供两个核心功能：

1. **自选股分析** — 用户添加关注的股票（A 股/港股），系统批量预测未来趋势，结合技术面、资金面多维评分排优先级
2. **主动选股** — 扫描沪深 300 + 中证 500 成分股，发现高回报且风险可控的买入标的

系统以独立 Flask Web 应用形式运行，不侵入 Kronos 现有代码。

> 参考了 clawhub 上的 [china-stock-analysis](https://clawhub.ai/paulshe/china-stock-analysis) 和 [akshare-stock](https://clawhub.ai/sunshine-del-ux/new-akshare-stock) 两个 skill 的设计。

## 技术决策

- **独立 Flask 应用**（方案 C）：新建 `app/` 目录，与现有 `webui/` 独立
- **理由**：不侵入项目原有代码，结构干净，技术栈一致（Python + Flask），可独立部署
- **数据源**：akshare（A 股/港股行情、资金流向、融资融券、板块等数据）
- **模型**：复用 Kronos 现有的 KronosPredictor API，支持 batch 预测
- **预测周期**：中线（20-60 个交易日）
- **风控策略**：Kronos 预测回撤 + 多样本不确定性 + 传统技术指标 + 资金面信号

## 目录结构

```
app/
├── app.py                  # Flask 应用入口 + 路由
├── config.py               # 配置项（模型路径、扫描范围、评分权重等）
├── services/
│   ├── data_fetcher.py     # 数据获取层（akshare 封装：行情/资金流向/融资融券/板块）
│   ├── predictor.py        # Kronos 预测封装（单例模型加载）
│   ├── technical.py        # 技术指标计算（均线、支撑压力位、成交量趋势）
│   ├── analyzer.py         # 综合分析与评分引擎
│   └── scanner.py          # 主动选股扫描器
├── models/
│   └── stock.py            # 数据模型定义
├── templates/
│   ├── base.html           # 基础布局模板
│   ├── watchlist.html      # 自选股面板
│   ├── discover.html       # 发现面板
│   └── detail.html         # 个股详情页（含分析摘要）
├── static/
│   ├── css/style.css
│   └── js/main.js
└── data/
    └── watchlist.json      # 自选股持久化存储
```

## 核心模块设计

### 1. 数据获取层 — `services/data_fetcher.py`

封装 akshare 数据获取，提供统一接口。参考 akshare-stock skill 的 API 用法。

```python
class DataFetcher:
    # --- 基础行情 ---
    def get_stock_history(symbol: str, lookback: int = 400) -> pd.DataFrame:
        """获取单只股票历史日线数据（ak.stock_zh_a_hist）
        返回: DataFrame with columns [date, open, high, low, close, volume, amount]
        """

    def get_stock_realtime(symbol: str) -> dict:
        """获取实时行情快照（ak.stock_zh_a_spot_em）
        返回: {price, open, high, low, volume, change_pct, ...}
        """

    def get_stock_info(symbol: str) -> dict:
        """获取股票基本信息（名称、行业、市场等）"""

    def search_stock(keyword: str) -> List[dict]:
        """搜索股票（代码或名称模糊匹配）"""

    # --- 指数成分股 ---
    def get_index_components(index: str = "hs300") -> List[str]:
        """获取指数成分股列表
        支持: hs300（沪深300）、zz500（中证500）
        """

    # --- 资金流向（新增）---
    def get_fund_flow(symbol: str, days: int = 5) -> pd.DataFrame:
        """获取个股资金流向（ak.stock_individual_fund_flow）
        返回: DataFrame with columns [date, main_net_inflow, retail_net_inflow, ...]
        主力净流入/散户净流入，近 N 日
        """

    # --- 融资融券（新增）---
    def get_margin_data(symbol: str, days: int = 20) -> pd.DataFrame:
        """获取融资融券数据（ak.stock_margin_detail_info）
        返回: DataFrame with columns [date, margin_balance, short_balance, ...]
        """

    # --- 板块信息（新增）---
    def get_stock_sector(symbol: str) -> dict:
        """获取个股所属行业和概念板块
        返回: {industry: str, concepts: List[str]}
        """

    def get_sector_trend(sector_name: str) -> dict:
        """获取板块近期涨跌趋势
        返回: {name, change_5d, change_20d, rank}
        """
```

**数据缓存**：获取的历史数据按日缓存到 `app/data/cache/` 目录，避免重复请求。缓存当日有效，次日自动失效。资金流向和融资融券数据缓存 2 小时。

### 2. 预测封装 — `services/predictor.py`

对 Kronos 模型的单例封装。

```python
class StockPredictor:
    _instance = None  # 单例

    def __init__(self, model_name="NeoQuasar/Kronos-small",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-base"):
        """加载模型和 tokenizer，仅初始化一次"""

    def predict_single(self, df: pd.DataFrame, pred_len: int = 40,
                       sample_count: int = 5) -> dict:
        """单只股票预测
        Returns: {
            "predictions": List[pd.DataFrame],  # sample_count 条预测路径
            "mean_pred": pd.DataFrame,           # 均值预测
            "std_pred": pd.DataFrame,            # 标准差
        }
        """

    def predict_batch(self, df_list: List[pd.DataFrame],
                      pred_len: int = 40, sample_count: int = 5) -> List[dict]:
        """批量预测，利用 KronosPredictor.predict_batch"""
```

**关键参数**：
- `pred_len`: 40 个交易日（约 2 个月），中线预测
- `sample_count`: 5 次采样，用于计算不确定性
- `T=1.0, top_p=0.9`: 与项目默认参数一致

**模型选择**：默认使用 `Kronos-small`（24.7M 参数），CPU 推理可接受。用户可在配置中切换为 `Kronos-base` 以获得更好效果。

### 3. 技术指标计算 — `services/technical.py`（新增）

参考 china-stock-analysis skill 的技术面分析维度，基于历史 K 线数据计算传统技术指标。

```python
class TechnicalAnalyzer:
    def analyze(self, df: pd.DataFrame) -> TechnicalResult:
        """计算技术指标并给出技术面评分
        Returns: TechnicalResult(
            ma_alignment,       # 均线排列: bullish/bearish/neutral
            ma_score,           # 均线评分 0-100
            support_level,      # 关键支撑位
            resistance_level,   # 关键压力位
            volume_trend,       # 成交量趋势: expanding/shrinking/stable
            volume_score,       # 量能评分 0-100
            technical_score,    # 技术面综合评分 0-100
            signals,            # 信号列表 List[str]
        )
        """
```

**具体指标**：

| 指标 | 计算方式 | 信号判定 |
|------|---------|---------|
| MA 均线排列 | MA5 > MA10 > MA20 > MA60 | 多头排列→看涨, 空头排列→看跌 |
| 价格与均线关系 | 收盘价相对 MA20 的位置 | 站上 MA20→看涨, 跌破→看跌 |
| 关键支撑/压力位 | 近 60 日的局部极值 | 接近支撑位→买入机会, 接近压力位→注意风险 |
| 成交量趋势 | 近 5 日均量 vs 近 20 日均量 | 放量上涨→积极信号, 缩量下跌→观望 |
| 量价配合 | 价格上涨+成交量放大 | 量价齐升→确认信号 |

**技术面评分（满分 100）**：
- 均线排列分（40%）：完美多头→100, 空头→0, 过渡→50
- 量能配合分（30%）：量价齐升→100, 量价背离→0
- 支撑压力位分（30%）：远离压力位且有支撑→100, 逼近压力位→0

### 4. 综合分析评分引擎 — `services/analyzer.py`

将 Kronos 预测 + 技术指标 + 资金面 + 融资融券整合为综合评分。

```python
class StockAnalyzer:
    def analyze(self, symbol: str, history_df: pd.DataFrame,
                prediction_result: dict) -> AnalysisResult:
        """综合分析单只股票
        整合: Kronos预测 + 技术面 + 资金面 + 融资融券
        Returns: AnalysisResult(...)
        """

    def generate_summary(self, result: AnalysisResult) -> str:
        """生成结构化分析摘要文案（参考 china-stock-analysis 输出格式）"""
```

**综合评分模型（满分 100）— 6 维度**：

| 维度 | 权重 | 数据来源 | 计算方式 | 评分映射 |
|------|------|---------|---------|---------|
| 预期收益率 | 30% | Kronos 预测 | `(pred_close[-1] / current_price - 1) * 100` | 0%→0, 5%→50, 10%→80, 20%+→100 |
| 最大回撤 | 20% | Kronos 预测 | 预测均值曲线的峰谷最大回撤 | 0%→100, 5%→80, 10%→50, 20%+→0 |
| 预测确定性 | 15% | Kronos 多样本 | `mean(std_pred['close']) / current_price` 归一化 | 低方差→100, 高方差→0 |
| 技术面 | 15% | 技术指标 | 均线排列 + 量能配合 + 支撑压力位 | 由 TechnicalAnalyzer 输出 |
| 资金面 | 10% | 资金流向 | 近 5 日主力净流入占流通市值比 | 净流入→加分, 净流出→减分 |
| 融资趋势 | 10% | 融资融券 | 近 20 日融资余额变化趋势 | 融资增→看多, 融资减→看空 |

**风险等级判定**：

| 等级 | 条件 |
|------|------|
| 低风险 (low) | 最大回撤 < 5% 且 确定性分 ≥ 70 且 技术面分 ≥ 60 |
| 中风险 (medium) | 最大回撤 5-15% 或 确定性分 40-70 |
| 高风险 (high) | 最大回撤 > 15% 或 确定性分 < 40 或 技术面分 < 30 |

**推荐等级**（参考 china-stock-analysis 的买入/持有/卖出框架）：

| 推荐 | 条件 | 信号特征 |
|------|------|---------|
| 强烈买入 (strong_buy) | 综合分 ≥ 80 且 风险 ≤ medium | 预测看涨 + 均线多头 + 主力净流入 |
| 买入 (buy) | 综合分 ≥ 65 且 风险 ≤ medium | 预测看涨 + 至少 2 个辅助信号确认 |
| 持有 (hold) | 综合分 40-65 | 趋势不明确 或 等待关键突破 |
| 回避 (avoid) | 综合分 < 40 或 风险 = high 且 收益低 | 预测看跌 + 均线空头 + 资金流出 |

**结构化分析摘要**（详情页展示，参考 china-stock-analysis 输出格式）：

```
## {股票名称}({代码}) 投资分析

### 核心数据
| 指标 | 数值 |
|------|------|
| 当前价 | 103.70 |
| 涨跌幅 | -1.74% |
| 成交量 | 42.68 万手 |
| 所属行业 | 汽车整车 |
| 板块热度 | 行业排名 12/68 |

### Kronos 预测
- 预测周期: 40 个交易日
- 预期收益率: +12.3%
- 预测最大回撤: 3.2%
- 预测确定性: 高（多样本标准差 2.1%）

### 技术面分析
- 均线状态: MA5 > MA10 > MA20，多头排列
- 关键支撑位: 100.50
- 关键压力位: 108.30
- 成交量: 近 5 日缩量整理

### 资金面分析
- 近 5 日主力净流入: +2.3 亿
- 融资余额趋势: 近 20 日增长 5.2%

### 投资建议
**建议: 买入** (综合评分 ★78)
- 理由: Kronos 模型预测未来 2 个月收益率 +12.3%，均线多头排列确认上升趋势，主力资金持续净流入
- 操作策略: 可在回调至支撑位 100.50 附近分批建仓
- 风险提示: 预测基于历史 K 线模式，不构成投资建议。关注压力位 108.30 的突破情况

> 分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。
```

### 5. 主动选股扫描器 — `services/scanner.py`

```python
class StockScanner:
    def scan(self, index_list=["hs300", "zz500"],
             top_n: int = 20) -> List[AnalysisResult]:
        """扫描指数成分股，返回 Top N 推荐
        流程:
        1. 获取成分股列表
        2. 批量获取历史数据
        3. 分批 batch 预测（每批 16 只）
        4. 逐只计算技术指标 + 资金面
        5. 综合评分
        6. 过滤: 综合分 >= 65 且 风险 <= medium
        7. 按综合分降序返回 Top N
        """

    def get_scan_progress(self) -> dict:
        """获取扫描进度（用于前端进度条）
        Returns: {"total": int, "completed": int, "status": str}
        """
```

**性能优化**：
- 分批预测：每批 16 只股票，利用 `predict_batch` 的 batch 能力
- 技术指标计算纯 pandas 操作，性能开销极小
- 资金流向/融资融券数据在扫描模式下简化（仅获取关键汇总值）
- 缓存机制：扫描结果缓存 4 小时，避免频繁重复扫描
- 进度追踪：异步扫描，前端轮询进度

### 6. 数据模型 — `models/stock.py`

```python
@dataclass
class WatchlistItem:
    symbol: str          # 股票代码
    name: str            # 股票名称
    market: str          # a_share / hk
    added_at: str        # 添加时间

@dataclass
class TechnicalResult:
    ma_alignment: str           # bullish/bearish/neutral
    ma_score: float             # 0-100
    support_level: float        # 支撑位
    resistance_level: float     # 压力位
    volume_trend: str           # expanding/shrinking/stable
    volume_score: float         # 0-100
    technical_score: float      # 0-100
    signals: List[str]          # 信号列表

@dataclass
class FundFlowResult:
    main_net_inflow_5d: float   # 近 5 日主力净流入（元）
    fund_flow_score: float      # 0-100

@dataclass
class MarginResult:
    margin_balance_trend: float # 近 20 日融资余额变化率 %
    margin_score: float         # 0-100

@dataclass
class AnalysisResult:
    symbol: str
    name: str
    industry: str               # 所属行业（新增）
    current_price: float
    change_pct: float           # 今日涨跌幅（新增）
    score: float                # 综合评分 0-100
    expected_return: float      # 预期收益率 %
    max_drawdown: float         # 最大回撤 %
    uncertainty: float          # 不确定性
    technical: TechnicalResult  # 技术面结果（新增）
    fund_flow: FundFlowResult   # 资金面结果（新增）
    margin: MarginResult        # 融资融券结果（新增）
    risk_level: str             # low/medium/high
    recommendation: str         # strong_buy/buy/hold/avoid
    summary: str                # 结构化分析摘要文案（新增）
    pred_prices: List[float]    # 预测收盘价序列
    pred_std: List[float]       # 预测标准差序列（用于不确定性区间）
    analyzed_at: str            # 分析时间
```

**持久化**：自选股列表以 JSON 存储在 `app/data/watchlist.json`。分析结果不持久化，每次按需计算。

## 页面设计

### 自选股面板 (`/watchlist`)

```
┌─────────────────────────────────────────────────────────────┐
│  Kronos Stock Advisor                                       │
│  [自选股]  [发现]                                            │
├─────────────────────────────────────────────────────────────┤
│  搜索添加股票: [________] [添加]                              │
├─────────────────────────────────────────────────────────────┤
│  排序: 综合评分 ▼  | 预期收益 | 风险等级                       │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 002594 比亚迪  汽车整车   103.70   +12.3%   ★85   │     │
│  │ [低风险] [强烈买入]  均线多头 | 主力净流入           │     │
│  ├────────────────────────────────────────────────────┤     │
│  │ 600519 贵州茅台  白酒   1580.00   +5.1%    ★72    │     │
│  │ [低风险] [买入]      均线中性 | 融资增长             │     │
│  ├────────────────────────────────────────────────────┤     │
│  │ 300750 宁德时代  电池   198.50    +8.7%    ★68    │     │
│  │ [中风险] [买入]      均线多头 | 资金流出             │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 发现面板 (`/discover`)

```
┌─────────────────────────────────────────────────────────────┐
│  [自选股]  [发现]                                            │
├─────────────────────────────────────────────────────────────┤
│  [开始扫描沪深300+中证500]                                    │
│  筛选: 风险[全部▼] 收益率[>5%▼] 行业[全部▼]                   │
│                                                             │
│  扫描完成 800/800  Top 20 推荐:                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │ #1  000858 五粮液  白酒   ★91  +15.2% 低风险       │     │
│  │     均线多头 | 主力净流入 3.1亿 | 融资增长            │     │
│  │ #2  601318 中国平安 保险  ★87  +11.8% 低风险        │     │
│  │     均线多头 | 主力净流入 1.8亿 | 融资增长            │     │
│  │ #3  ...                                            │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 个股详情页 (`/detail/<symbol>`)

- **预测 K 线图**（Plotly）：历史蓝线 + 预测红线 + 不确定性区间（灰色阴影）+ 支撑/压力位水平线
- **评分雷达图**：6 个维度的拆解（预期收益、回撤、确定性、技术面、资金面、融资趋势）
- **核心数据表**：价格、涨跌幅、成交量、行业、板块热度
- **分析摘要卡片**：结构化文案（Kronos 预测 + 技术面 + 资金面 + 投资建议）
- **资金流向图表**：近 5/10/20 日主力净流入柱状图
- **操作按钮**：添加到自选股 / 刷新预测

## API 路由

| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 重定向到 `/watchlist` |
| `/watchlist` | GET | 自选股面板 |
| `/discover` | GET | 发现面板 |
| `/detail/<symbol>` | GET | 个股详情页 |
| `/api/watchlist` | GET | 获取自选股列表 |
| `/api/watchlist` | POST | 添加自选股 `{symbol}` |
| `/api/watchlist/<symbol>` | DELETE | 删除自选股 |
| `/api/analyze/<symbol>` | GET | 分析单只股票（含全维度） |
| `/api/analyze/batch` | POST | 批量分析 `{symbols: [...]}` |
| `/api/scan/start` | POST | 启动扫描 |
| `/api/scan/progress` | GET | 获取扫描进度 |
| `/api/scan/results` | GET | 获取扫描结果 |
| `/api/search` | GET | 搜索股票 `?q=keyword` |

## 配置项 — `config.py`

```python
# 模型配置
MODEL_NAME = "NeoQuasar/Kronos-small"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512
DEVICE = None  # auto-detect

# 预测配置
PRED_LEN = 40          # 预测 40 个交易日（约 2 个月）
SAMPLE_COUNT = 5       # 多次采样用于不确定性估计
TEMPERATURE = 1.0
TOP_P = 0.9
LOOKBACK = 400         # 历史回看天数

# 评分权重（6 维度）
SCORE_WEIGHTS = {
    "expected_return": 0.30,
    "max_drawdown": 0.20,
    "uncertainty": 0.15,
    "technical": 0.15,
    "fund_flow": 0.10,
    "margin": 0.10,
}

# 扫描配置
SCAN_INDICES = ["hs300", "zz500"]
SCAN_BATCH_SIZE = 16   # 每批预测股票数
SCAN_TOP_N = 20        # 返回 Top N 推荐
SCAN_CACHE_HOURS = 4   # 扫描结果缓存时间

# 涨跌停限制（A 股 10%）
PRICE_LIMIT_RATE = 0.10

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001      # 避免与现有 webui 冲突
```

## 依赖

在现有 `requirements.txt` 基础上额外需要：
- `flask` + `flask-cors`（Web 框架）
- `akshare`（A 股/港股行情、资金流向、融资融券、板块数据）
- `plotly`（交互式图表）

## 运行方式

```bash
cd Kronos
python -m app.app
# 访问 http://localhost:5001
```

## 限制与免责

- 所有预测和评分仅供研究参考，**不构成投资建议**。投资有风险，入市需谨慎。
- 模型预测基于历史 K 线模式，无法预测突发事件（政策、黑天鹅等）
- 技术指标和资金面数据为辅助参考，不能单独作为买卖依据
- 扫描 800 只股票在 CPU 上耗时较长，建议有 GPU 环境
- 港股支持取决于 akshare 的数据覆盖范围
- 资金流向和融资融券接口可能因数据源变动而失效，建议添加异常处理
