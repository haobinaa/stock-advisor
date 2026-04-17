"""Technical analysis service for calculating traditional indicators from K-line data."""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.models.stock import TechnicalResult

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Calculates technical indicators from historical K-line data."""

    # MA periods
    MA_PERIODS = [5, 10, 20, 60]

    # Weights for final technical_score
    WEIGHT_MA = 0.4
    WEIGHT_SUPPORT_RESISTANCE = 0.3
    WEIGHT_VOLUME = 0.3

    def analyze(self, df: pd.DataFrame) -> TechnicalResult:
        """Analyze technical indicators from historical K-line data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume],
                at least 60 rows, sorted by date ascending.

        Returns:
            TechnicalResult with all indicator scores and signals.
        """
        if len(df) < 60:
            logger.warning("DataFrame has %d rows, expected at least 60", len(df))

        signals: List[str] = []

        # Calculate moving averages
        ma_values = self._calc_moving_averages(df)
        ma_alignment, ma_score = self._evaluate_ma_alignment(df, ma_values)
        signals.extend(self._ma_signals(ma_alignment))

        # Support and resistance levels
        support, resistance, sr_score = self._calc_support_resistance(df)
        signals.extend(self._sr_signals(df, support, resistance))

        # Volume trend
        volume_trend, volume_score = self._calc_volume_trend(df)
        signals.extend(self._volume_signals(df, volume_trend))

        # Weighted technical score
        technical_score = (
            ma_score * self.WEIGHT_MA
            + sr_score * self.WEIGHT_SUPPORT_RESISTANCE
            + volume_score * self.WEIGHT_VOLUME
        )

        logger.info(
            "Technical analysis complete: ma_score=%.1f, sr_score=%.1f, "
            "vol_score=%.1f, total=%.1f",
            ma_score, sr_score, volume_score, technical_score,
        )

        return TechnicalResult(
            ma_alignment=ma_alignment,
            ma_score=ma_score,
            support_level=support,
            resistance_level=resistance,
            volume_trend=volume_trend,
            volume_score=volume_score,
            technical_score=round(technical_score, 2),
            signals=signals,
        )

    # ------------------------------------------------------------------ #
    #  Moving Average Alignment
    # ------------------------------------------------------------------ #

    def _calc_moving_averages(self, df: pd.DataFrame) -> dict[int, float]:
        """Return the latest MA value for each period."""
        ma_values: dict[int, float] = {}
        for period in self.MA_PERIODS:
            if len(df) >= period:
                ma_values[period] = float(df["close"].rolling(period).mean().iloc[-1])
            else:
                ma_values[period] = float(df["close"].mean())
        return ma_values

    def _evaluate_ma_alignment(
        self, df: pd.DataFrame, ma: dict[int, float]
    ) -> Tuple[str, float]:
        """Evaluate moving average alignment and return (alignment, score)."""
        ma5, ma10, ma20, ma60 = ma[5], ma[10], ma[20], ma[60]
        current_price = float(df["close"].iloc[-1])

        # Full bullish alignment: MA5 > MA10 > MA20 > MA60
        if ma5 > ma10 > ma20 > ma60:
            return "bullish", 100.0

        # Full bearish alignment: MA5 < MA10 < MA20 < MA60
        if ma5 < ma10 < ma20 < ma60:
            return "bearish", 0.0

        # Partial bullish: price above MA20
        if current_price > ma20:
            return "neutral", 70.0

        # Otherwise neutral / mixed signals
        return "neutral", 50.0

    @staticmethod
    def _ma_signals(alignment: str) -> List[str]:
        if alignment == "bullish":
            return ["均线多头排列"]
        if alignment == "bearish":
            return ["均线空头排列"]
        return ["均线交叉整理"]

    # ------------------------------------------------------------------ #
    #  Support / Resistance Levels
    # ------------------------------------------------------------------ #

    def _calc_support_resistance(
        self, df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Find support/resistance via rolling-window local extrema.

        Returns (support_level, resistance_level, score).
        """
        window = 5  # rolling window for local extrema detection
        lookback = min(60, len(df))
        recent = df.tail(lookback)
        current_price = float(df["close"].iloc[-1])

        # Local minima (support candidates)
        low_series = recent["low"]
        supports = self._find_local_minima(low_series, window)
        supports = [s for s in supports if s < current_price]

        # Local maxima (resistance candidates)
        high_series = recent["high"]
        resistances = self._find_local_maxima(high_series, window)
        resistances = [r for r in resistances if r > current_price]

        # Pick nearest support and resistance
        support = max(supports) if supports else current_price * 0.95
        resistance = min(resistances) if resistances else current_price * 1.05

        # Score: far from resistance + near support → high score
        price_range = resistance - support
        if price_range <= 0:
            score = 50.0
        else:
            # Position within the support-resistance range (0 = at support, 1 = at resistance)
            position = (current_price - support) / price_range
            # Near support (position ~0) → high score; near resistance (position ~1) → low score
            score = max(0.0, min(100.0, (1.0 - position) * 100.0))

        return round(support, 2), round(resistance, 2), round(score, 2)

    @staticmethod
    def _find_local_minima(series: pd.Series, window: int) -> List[float]:
        """Find local minima using a rolling window comparison."""
        values = series.values
        minima: List[float] = []
        for i in range(window, len(values) - window):
            left = values[i - window : i]
            right = values[i + 1 : i + window + 1]
            if values[i] <= left.min() and values[i] <= right.min():
                minima.append(float(values[i]))
        return minima

    @staticmethod
    def _find_local_maxima(series: pd.Series, window: int) -> List[float]:
        """Find local maxima using a rolling window comparison."""
        values = series.values
        maxima: List[float] = []
        for i in range(window, len(values) - window):
            left = values[i - window : i]
            right = values[i + 1 : i + window + 1]
            if values[i] >= left.max() and values[i] >= right.max():
                maxima.append(float(values[i]))
        return maxima

    @staticmethod
    def _sr_signals(
        df: pd.DataFrame, support: float, resistance: float
    ) -> List[str]:
        """Generate support/resistance proximity signals."""
        signals: List[str] = []
        current_price = float(df["close"].iloc[-1])
        threshold = 0.02  # 2% proximity threshold

        if support > 0 and (current_price - support) / current_price < threshold:
            signals.append(f"接近支撑位 {support:.2f}")
        if resistance > 0 and (resistance - current_price) / current_price < threshold:
            signals.append(f"接近压力位 {resistance:.2f}")

        return signals

    # ------------------------------------------------------------------ #
    #  Volume Trend
    # ------------------------------------------------------------------ #

    def _calc_volume_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Evaluate volume trend and price-volume alignment.

        Returns (trend_label, score).
        """
        vol_5d = float(df["volume"].tail(5).mean())
        vol_20d = float(df["volume"].tail(20).mean())

        if vol_20d == 0:
            logger.warning("20-day volume mean is 0, defaulting to stable")
            return "stable", 50.0

        ratio = vol_5d / vol_20d

        if ratio > 1.3:
            trend = "expanding"
        elif ratio < 0.7:
            trend = "shrinking"
        else:
            trend = "stable"

        # Price direction: compare last close vs close 5 days ago
        price_now = float(df["close"].iloc[-1])
        price_5d_ago = float(df["close"].iloc[-6]) if len(df) >= 6 else float(df["close"].iloc[0])
        price_up = price_now >= price_5d_ago

        # Score based on price-volume alignment
        if price_up and trend == "expanding":
            score = 100.0  # confirmed uptrend
        elif price_up and trend == "shrinking":
            score = 40.0  # divergence, weak
        elif not price_up and trend == "expanding":
            score = 30.0  # selling pressure
        elif not price_up and trend == "shrinking":
            score = 60.0  # selling exhaustion
        else:
            # stable volume
            score = 50.0

        return trend, score

    @staticmethod
    def _volume_signals(df: pd.DataFrame, trend: str) -> List[str]:
        """Generate volume-related signals."""
        price_now = float(df["close"].iloc[-1])
        price_5d_ago = float(df["close"].iloc[-6]) if len(df) >= 6 else float(df["close"].iloc[0])
        price_up = price_now >= price_5d_ago

        if price_up and trend == "expanding":
            return ["放量上涨"]
        if price_up and trend == "shrinking":
            return ["缩量上涨"]
        if not price_up and trend == "expanding":
            return ["放量下跌"]
        if not price_up and trend == "shrinking":
            return ["缩量整理"]
        return ["成交量平稳"]
