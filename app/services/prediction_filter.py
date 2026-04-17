import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PredictionFilter:
    """Filter and adjust predictions based on backtest statistics."""

    def __init__(self, backtest_stats: dict = None):
        self._stats = backtest_stats or {}
        self._industry_accuracy: Dict[str, float] = {}

        by_industry = self._stats.get("by_industry", {})
        for industry, info in by_industry.items():
            if isinstance(info, dict) and "accuracy" in info:
                self._industry_accuracy[industry] = info["accuracy"]

    def should_exclude(
        self, symbol: str, name: str, industry: str
    ) -> Tuple[bool, str]:
        """Decide whether a stock should be excluded from recommendations.

        Returns:
            (should_exclude, reason) tuple.
        """
        # ST / *ST stocks
        if "ST" in name or "*ST" in name:
            return True, "ST stock"

        # Delisting stocks
        if "退" in name:
            return True, "delisting"

        # Low industry accuracy
        if industry in self._industry_accuracy:
            acc = self._industry_accuracy[industry]
            if acc < 0.4:
                return True, f"low accuracy in {industry}"

        return False, ""

    def get_confidence_adjustment(self, industry: str) -> float:
        """Return a confidence multiplier (0.5-1.5) based on industry accuracy.

        Higher accuracy in backtests → higher confidence in predictions.
        """
        if industry not in self._industry_accuracy:
            return 1.0

        acc = self._industry_accuracy[industry]

        if acc >= 0.7:
            return 1.2
        elif acc >= 0.5:
            return 1.0
        elif acc >= 0.4:
            return 0.8
        else:
            return 0.5
