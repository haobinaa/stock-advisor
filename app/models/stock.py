from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WatchlistItem:
    symbol: str
    name: str
    market: str  # a_share / hk
    added_at: str


@dataclass
class TechnicalResult:
    ma_alignment: str  # bullish / bearish / neutral
    ma_score: float
    support_level: float
    resistance_level: float
    volume_trend: str  # expanding / shrinking / stable
    volume_score: float
    technical_score: float
    signals: List[str] = field(default_factory=list)


@dataclass
class FundFlowResult:
    main_net_inflow_5d: float  # yuan
    fund_flow_score: float


@dataclass
class MarginResult:
    margin_balance_trend: float  # % change over 20 days
    margin_score: float


@dataclass
class AnalysisResult:
    symbol: str
    name: str
    industry: str
    current_price: float
    change_pct: float
    score: float  # 0-100
    expected_return: float  # %
    max_drawdown: float  # %
    uncertainty: float
    technical: Optional[TechnicalResult] = None
    fund_flow: Optional[FundFlowResult] = None
    margin: Optional[MarginResult] = None
    risk_level: str = "medium"  # low / medium / high
    recommendation: str = "hold"  # strong_buy / buy / hold / avoid
    summary: str = ""
    pred_prices: List[float] = field(default_factory=list)
    pred_std: List[float] = field(default_factory=list)
    analyzed_at: str = ""


@dataclass
class PredictionRecord:
    id: str
    symbol: str
    name: str
    industry: str
    predicted_at: str  # ISO date
    pred_len: int
    current_price: float  # price at prediction time
    pred_prices: List[float]
    expected_return: float
    score: float
    recommendation: str
    risk_level: str
    # Dimension scores for weight optimization
    dim_scores: Optional[dict] = None
    # Filled after backtest
    backtest: Optional["BacktestResult"] = None


@dataclass
class BacktestResult:
    actual_prices: List[float]
    actual_return: float  # %
    predicted_return: float  # %
    return_error: float  # absolute diff
    direction_correct: bool  # both positive or both negative
    actual_max_drawdown: float
    backtested_at: str
