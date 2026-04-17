import json
import logging
from typing import Dict, List, Optional

import numpy as np

from app import config

logger = logging.getLogger(__name__)

# Dimension keys in fixed order matching config.SCORE_WEIGHTS
DIM_KEYS = list(config.SCORE_WEIGHTS.keys())

MIN_RECORDS = 20


class WeightOptimizer:
    """Optimize scoring weights via linear regression on backtest results."""

    def optimize(self, records: List[dict]) -> Optional[Dict[str, float]]:
        """Fit weights from historical prediction records.

        Args:
            records: list of prediction records containing both
                     ``dim_scores`` and ``backtest`` fields.

        Returns:
            Optimized weight dict (same keys as config.SCORE_WEIGHTS),
            or None if insufficient data.
        """
        # Filter to records that have both dim_scores and backtest
        valid = [
            r for r in records
            if r.get("dim_scores") and r.get("backtest")
        ]

        if len(valid) < MIN_RECORDS:
            logger.warning(
                "Not enough records for weight optimization: %d < %d",
                len(valid), MIN_RECORDS,
            )
            return None

        # Build feature matrix X and label vector Y
        X = np.array([
            [r["dim_scores"].get(k, 0.0) for k in DIM_KEYS]
            for r in valid
        ], dtype=np.float64)

        Y = np.array([
            r["backtest"]["actual_return"]
            for r in valid
        ], dtype=np.float64)

        # Least-squares solve: Y = X @ w
        w, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)

        # Clamp negative weights to 0
        w = np.maximum(w, 0.0)

        # Normalize so weights sum to 1.0
        total = w.sum()
        if total <= 0:
            logger.warning("All fitted weights <= 0, returning equal weights")
            w = np.ones(len(DIM_KEYS)) / len(DIM_KEYS)
        else:
            w = w / total

        weights = {k: float(round(v, 4)) for k, v in zip(DIM_KEYS, w)}
        logger.info("Optimized weights: %s", weights)
        return weights

    def evaluate(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        records: List[dict],
    ) -> dict:
        """Compare old vs new weights on historical records.

        Ranks stocks by weighted score, then compares actual returns of
        the top 20% vs bottom 20% under each weight set.
        """
        valid = [
            r for r in records
            if r.get("dim_scores") and r.get("backtest")
        ]

        def _score(rec: dict, weights: Dict[str, float]) -> float:
            return sum(
                rec["dim_scores"].get(k, 0.0) * weights.get(k, 0.0)
                for k in DIM_KEYS
            )

        def _evaluate_weights(weights: Dict[str, float]):
            scored = [
                (_score(r, weights), r["backtest"]["actual_return"])
                for r in valid
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            n = len(scored)
            top_n = max(1, n // 5)
            bot_n = max(1, n // 5)

            top_returns = [s[1] for s in scored[:top_n]]
            bot_returns = [s[1] for s in scored[-bot_n:]]

            top_return = float(np.mean(top_returns)) if top_returns else 0.0

            # R² calculation
            pred_scores = np.array([s[0] for s in scored])
            actual_returns = np.array([s[1] for s in scored])
            ss_res = np.sum((actual_returns - pred_scores) ** 2)
            ss_tot = np.sum((actual_returns - np.mean(actual_returns)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            return top_return, r2

        old_top, old_r2 = _evaluate_weights(old_weights)
        new_top, new_r2 = _evaluate_weights(new_weights)

        improvement = (
            ((new_top - old_top) / abs(old_top) * 100)
            if old_top != 0 else 0.0
        )

        result = {
            "old_top_return": round(old_top, 4),
            "new_top_return": round(new_top, 4),
            "old_r2": round(old_r2, 4),
            "new_r2": round(new_r2, 4),
            "improvement_pct": round(improvement, 2),
        }
        logger.info("Weight evaluation: %s", result)
        return result

    def save_weights(self, weights: Dict[str, float]) -> None:
        """Persist optimized weights to config.OPTIMIZED_WEIGHTS_FILE."""
        path = config.OPTIMIZED_WEIGHTS_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
        logger.info("Saved optimized weights to %s", path)

    def load_weights(self) -> Optional[Dict[str, float]]:
        """Load optimized weights from disk, or None if not found."""
        import os

        path = config.OPTIMIZED_WEIGHTS_FILE
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                weights = json.load(f)
            logger.info("Loaded optimized weights from %s", path)
            return weights
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load optimized weights: %s", e)
            return None
