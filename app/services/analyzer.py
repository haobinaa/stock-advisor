import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from app import config
from app.config import SCORE_WEIGHTS
from app.models.stock import (
    AnalysisResult,
    FundFlowResult,
    MarginResult,
    TechnicalResult,
)

logger = logging.getLogger(__name__)


class StockAnalyzer:

    def __init__(self):
        self._weights = dict(SCORE_WEIGHTS)
        self._load_optimized_weights()

    def _load_optimized_weights(self):
        """Load optimized weights from file if available."""
        try:
            path = config.OPTIMIZED_WEIGHTS_FILE
            if os.path.exists(path):
                with open(path, "r") as f:
                    loaded = json.load(f)
                if set(loaded.keys()) == set(SCORE_WEIGHTS.keys()):
                    self._weights = loaded
                    logger.info("Loaded optimized weights: %s", loaded)
        except Exception as e:
            logger.warning("Failed to load optimized weights: %s", e)

    @property
    def current_weights(self) -> dict:
        return dict(self._weights)

    def reload_weights(self):
        """Reload weights from disk."""
        self._weights = dict(SCORE_WEIGHTS)
        self._load_optimized_weights()
        self._last_dim_scores = {}

    def get_last_dim_scores(self) -> dict:
        """Return the raw dimension scores from the last analyze() call."""
        return dict(self._last_dim_scores)

    def _score_expected_return(self, pct: float) -> float:
        if pct <= 0:
            return max(0, 50 + pct * 5)  # negative returns penalized
        if pct <= 5:
            return pct * 10  # 0-5% → 0-50
        if pct <= 10:
            return 50 + (pct - 5) * 6  # 5-10% → 50-80
        if pct <= 20:
            return 80 + (pct - 10) * 2  # 10-20% → 80-100
        return 100

    def _score_max_drawdown(self, dd_pct: float) -> float:
        dd = abs(dd_pct)
        if dd <= 0:
            return 100
        if dd <= 5:
            return 100 - dd * 4  # 0-5% → 100-80
        if dd <= 10:
            return 80 - (dd - 5) * 6  # 5-10% → 80-50
        if dd <= 20:
            return 50 - (dd - 10) * 5  # 10-20% → 50-0
        return 0

    def _score_uncertainty(self, rel_std: float) -> float:
        # rel_std = mean(std) / price, typically 0.01-0.15
        if rel_std <= 0.02:
            return 100
        if rel_std <= 0.05:
            return 100 - (rel_std - 0.02) * 1000  # 0.02-0.05 → 100-70
        if rel_std <= 0.10:
            return 70 - (rel_std - 0.05) * 800  # 0.05-0.10 → 70-30
        return max(0, 30 - (rel_std - 0.10) * 300)

    def _calc_max_drawdown(self, prices: list) -> float:
        if not prices or len(prices) < 2:
            return 0.0
        arr = np.array(prices)
        peak = arr[0]
        max_dd = 0.0
        for p in arr[1:]:
            if p > peak:
                peak = p
            dd = (peak - p) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _determine_risk_level(
        self, max_dd: float, uncertainty_score: float,
        technical_score: float
    ) -> str:
        if max_dd < 5 and uncertainty_score >= 70 and technical_score >= 60:
            return "low"
        if max_dd > 15 or uncertainty_score < 40 or technical_score < 30:
            return "high"
        return "medium"

    def _determine_recommendation(
        self, score: float, risk_level: str, expected_return: float
    ) -> str:
        if score >= 80 and risk_level in ("low", "medium"):
            return "strong_buy"
        if score >= 65 and risk_level in ("low", "medium"):
            return "buy"
        if score < 40 or (risk_level == "high" and expected_return < 5):
            return "avoid"
        return "hold"

    def analyze(
        self,
        symbol: str,
        name: str,
        industry: str,
        current_price: float,
        change_pct: float,
        mean_pred: pd.DataFrame,
        std_pred: pd.DataFrame,
        technical: Optional[TechnicalResult] = None,
        fund_flow: Optional[FundFlowResult] = None,
        margin: Optional[MarginResult] = None,
        confidence_adjustment: float = 1.0,
    ) -> AnalysisResult:

        pred_closes = mean_pred["close"].tolist()
        std_closes = std_pred["close"].tolist() if std_pred is not None else []

        # --- Kronos prediction scores ---
        if pred_closes:
            final_price = pred_closes[-1]
            expected_return = (final_price / current_price - 1) * 100
        else:
            expected_return = 0.0

        max_drawdown = self._calc_max_drawdown(pred_closes)

        if std_closes and current_price > 0:
            rel_std = float(np.mean(std_closes)) / current_price
        else:
            rel_std = 0.05  # default moderate uncertainty

        score_return = self._score_expected_return(expected_return)
        score_dd = self._score_max_drawdown(max_drawdown)
        score_unc = self._score_uncertainty(rel_std)

        # --- Technical score ---
        tech_score = technical.technical_score if technical else 50.0

        # --- Fund flow score ---
        ff_score = fund_flow.fund_flow_score if fund_flow else 50.0

        # --- Margin score ---
        mg_score = margin.margin_score if margin else 50.0

        # --- Store raw dimension scores for weight optimization ---
        self._last_dim_scores = {
            "expected_return": round(score_return, 2),
            "max_drawdown": round(score_dd, 2),
            "uncertainty": round(score_unc, 2),
            "technical": round(tech_score, 2),
            "fund_flow": round(ff_score, 2),
            "margin": round(mg_score, 2),
        }

        # --- Composite score (use potentially optimized weights) ---
        w = self._weights
        composite = (
            score_return * w["expected_return"]
            + score_dd * w["max_drawdown"]
            + score_unc * w["uncertainty"]
            + tech_score * w["technical"]
            + ff_score * w["fund_flow"]
            + mg_score * w["margin"]
        )

        # Apply confidence adjustment from prediction filter
        composite = composite * confidence_adjustment
        composite = round(min(100, max(0, composite)), 1)

        risk_level = self._determine_risk_level(max_drawdown, score_unc, tech_score)
        recommendation = self._determine_recommendation(
            composite, risk_level, expected_return
        )

        summary = self.generate_summary(
            symbol=symbol,
            name=name,
            industry=industry,
            current_price=current_price,
            change_pct=change_pct,
            expected_return=expected_return,
            max_drawdown=max_drawdown,
            rel_std=rel_std,
            composite=composite,
            risk_level=risk_level,
            recommendation=recommendation,
            technical=technical,
            fund_flow=fund_flow,
            margin=margin,
        )

        return AnalysisResult(
            symbol=symbol,
            name=name,
            industry=industry,
            current_price=current_price,
            change_pct=change_pct,
            score=composite,
            expected_return=round(expected_return, 2),
            max_drawdown=round(max_drawdown, 2),
            uncertainty=round(rel_std * 100, 2),
            technical=technical,
            fund_flow=fund_flow,
            margin=margin,
            risk_level=risk_level,
            recommendation=recommendation,
            summary=summary,
            pred_prices=pred_closes,
            pred_std=std_closes,
            analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def generate_summary(
        self, symbol, name, industry, current_price, change_pct,
        expected_return, max_drawdown, rel_std, composite,
        risk_level, recommendation,
        technical, fund_flow, margin,
    ) -> str:
        rec_map = {
            "strong_buy": "强烈买入",
            "buy": "买入",
            "hold": "持有观望",
            "avoid": "建议回避",
        }
        risk_map = {"low": "低", "medium": "中", "high": "高"}

        lines = []
        lines.append(f"## {name}({symbol}) 投资分析\n")

        # Core data
        lines.append("### 核心数据")
        lines.append(f"- 当前价: {current_price:.2f}")
        lines.append(f"- 涨跌幅: {change_pct:+.2f}%")
        if industry:
            lines.append(f"- 所属行业: {industry}")
        lines.append("")

        # Kronos prediction
        lines.append("### Kronos 预测")
        lines.append(f"- 预测周期: 40 个交易日")
        lines.append(f"- 预期收益率: {expected_return:+.1f}%")
        lines.append(f"- 预测最大回撤: {max_drawdown:.1f}%")
        certainty = "高" if rel_std < 0.03 else ("中" if rel_std < 0.08 else "低")
        lines.append(f"- 预测确定性: {certainty}")
        lines.append("")

        # Technical
        if technical:
            lines.append("### 技术面分析")
            ma_map = {"bullish": "多头排列", "bearish": "空头排列", "neutral": "中性"}
            lines.append(f"- 均线状态: {ma_map.get(technical.ma_alignment, '未知')}")
            lines.append(f"- 关键支撑位: {technical.support_level:.2f}")
            lines.append(f"- 关键压力位: {technical.resistance_level:.2f}")
            vol_map = {"expanding": "放量", "shrinking": "缩量", "stable": "平稳"}
            lines.append(
                f"- 成交量: {vol_map.get(technical.volume_trend, '未知')}"
            )
            lines.append("")

        # Fund flow
        if fund_flow:
            lines.append("### 资金面分析")
            inflow = fund_flow.main_net_inflow_5d
            if abs(inflow) >= 1e8:
                lines.append(f"- 近 5 日主力净流入: {inflow / 1e8:+.1f} 亿")
            else:
                lines.append(f"- 近 5 日主力净流入: {inflow / 1e4:+.0f} 万")

        # Margin
        if margin:
            trend = margin.margin_balance_trend
            if trend != 0:
                lines.append(f"- 融资余额趋势: 近 20 日{'增长' if trend > 0 else '下降'} {abs(trend):.1f}%")
            lines.append("")

        # Recommendation
        lines.append("### 投资建议")
        lines.append(
            f"**建议: {rec_map.get(recommendation, recommendation)}** "
            f"(综合评分 ★{composite:.0f}, 风险{risk_map.get(risk_level, risk_level)})"
        )

        # Reasoning
        reasons = []
        if expected_return > 5:
            reasons.append(f"Kronos 模型预测未来 2 个月收益率 {expected_return:+.1f}%")
        if technical and technical.ma_alignment == "bullish":
            reasons.append("均线多头排列确认上升趋势")
        if fund_flow and fund_flow.main_net_inflow_5d > 0:
            reasons.append("主力资金持续净流入")
        if margin and margin.margin_balance_trend > 0:
            reasons.append("融资余额增长，市场看多情绪")

        if reasons:
            lines.append(f"- 理由: {'; '.join(reasons)}")

        # Risk warning
        warnings = []
        if max_drawdown > 10:
            warnings.append(f"预测回撤较大({max_drawdown:.1f}%)")
        if technical and technical.ma_alignment == "bearish":
            warnings.append("均线空头排列")
        if fund_flow and fund_flow.main_net_inflow_5d < 0:
            warnings.append("主力资金流出")

        if warnings:
            lines.append(f"- 风险提示: {'; '.join(warnings)}")
        else:
            lines.append("- 风险提示: 注意仓位控制")

        lines.append("")
        lines.append("> 分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。")

        return "\n".join(lines)
