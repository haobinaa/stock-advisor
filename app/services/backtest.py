"""Prediction logging and backtesting service for Kronos Stock Advisor."""

import json
import logging
import uuid
from dataclasses import asdict
from datetime import date, datetime
from typing import List

import pandas as pd

from app import config
from app.models.stock import BacktestResult, PredictionRecord

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Log predictions and run backtests against actual market data."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_prediction(self, analysis_result, dim_scores: dict = None) -> str:
        """Create a PredictionRecord from an AnalysisResult and append to JSONL file.

        Returns the generated prediction id.
        """
        pred_id = uuid.uuid4().hex[:12]
        record = PredictionRecord(
            id=pred_id,
            symbol=analysis_result.symbol,
            name=analysis_result.name,
            industry=analysis_result.industry,
            predicted_at=date.today().strftime("%Y-%m-%d"),
            pred_len=config.PRED_LEN,
            current_price=analysis_result.current_price,
            pred_prices=analysis_result.pred_prices,
            expected_return=analysis_result.expected_return,
            score=analysis_result.score,
            recommendation=analysis_result.recommendation,
            risk_level=analysis_result.risk_level,
            dim_scores=dim_scores,
            backtest=None,
        )

        line = json.dumps(asdict(record), ensure_ascii=False)
        with open(config.PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        logger.info("Logged prediction %s for %s (%s)", pred_id, record.symbol, record.name)
        return pred_id

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_all_records(self) -> List[dict]:
        """Read all prediction records from JSONL file."""
        try:
            with open(config.PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            logger.debug("Predictions file not found: %s", config.PREDICTIONS_FILE)
            return []

    def get_pending_backtests(self) -> List[dict]:
        """Return records whose prediction window has expired but haven't been backtested."""
        today = date.today()
        pending = []
        for record in self.get_all_records():
            if record.get("backtest") is not None:
                continue
            predicted_at = datetime.strptime(record["predicted_at"], "%Y-%m-%d").date()
            pred_len = record.get("pred_len", config.PRED_LEN)
            # Calculate expiry date as pred_len business days after predicted_at
            bdays = pd.bdate_range(start=predicted_at, periods=pred_len + 1)
            expiry = bdays[-1].date()
            if expiry < today:
                pending.append(record)
        return pending

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def run_backtest(self, record: dict, data_fetcher) -> dict:
        """Run backtest for a single prediction record.

        Uses data_fetcher.get_stock_history(symbol) to obtain actual prices.
        Returns the updated record with backtest results filled in.
        """
        symbol = record["symbol"]
        predicted_at = datetime.strptime(record["predicted_at"], "%Y-%m-%d").date()
        pred_len = record.get("pred_len", config.PRED_LEN)

        # Business day range covering the prediction window
        bdays = pd.bdate_range(start=predicted_at, periods=pred_len + 1)
        start_date = bdays[0].strftime("%Y-%m-%d")
        end_date = bdays[-1].strftime("%Y-%m-%d")

        logger.info("Backtesting %s (%s) from %s to %s", symbol, record["name"], start_date, end_date)

        history = data_fetcher.get_stock_history(symbol, start=start_date, end=end_date)
        actual_prices = history["close"].tolist() if hasattr(history, "close") else []

        if not actual_prices:
            logger.warning("No price data for %s in backtest period", symbol)
            return record

        # Metrics
        current_price = record["current_price"]
        actual_return = (actual_prices[-1] / current_price - 1) * 100
        predicted_return = record["expected_return"]
        return_error = abs(actual_return - predicted_return)
        direction_correct = (actual_return >= 0) == (predicted_return >= 0)

        # Max drawdown: peak-to-trough
        actual_max_drawdown = self._calc_max_drawdown(actual_prices)

        backtest = BacktestResult(
            actual_prices=actual_prices,
            actual_return=round(actual_return, 4),
            predicted_return=round(predicted_return, 4),
            return_error=round(return_error, 4),
            direction_correct=direction_correct,
            actual_max_drawdown=round(actual_max_drawdown, 4),
            backtested_at=date.today().strftime("%Y-%m-%d"),
        )

        record["backtest"] = asdict(backtest)
        self._update_record(record)

        logger.info(
            "Backtest done for %s: actual=%.2f%% predicted=%.2f%% direction=%s",
            symbol, actual_return, predicted_return, direction_correct,
        )
        return record

    def run_all_pending(self, data_fetcher) -> int:
        """Run backtests on all pending records. Returns count of backtested records."""
        pending = self.get_pending_backtests()
        if not pending:
            logger.info("No pending backtests")
            return 0

        count = 0
        for record in pending:
            try:
                self.run_backtest(record, data_fetcher)
                count += 1
            except Exception:
                logger.exception("Failed to backtest %s", record.get("symbol"))
        logger.info("Backtested %d / %d pending records", count, len(pending))
        return count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Compute aggregate statistics from backtested prediction records."""
        all_records = self.get_all_records()
        backtested = [r for r in all_records if r.get("backtest")]

        stats: dict = {
            "total_predictions": len(all_records),
            "backtested_count": len(backtested),
            "direction_accuracy": 0.0,
            "avg_return_error": 0.0,
            "avg_predicted_return": 0.0,
            "avg_actual_return": 0.0,
            "by_recommendation": {},
            "by_industry": {},
            "by_score_range": {},
        }

        if not backtested:
            return stats

        direction_hits = sum(1 for r in backtested if r["backtest"]["direction_correct"])
        stats["direction_accuracy"] = round(direction_hits / len(backtested), 4)
        stats["avg_return_error"] = round(
            sum(r["backtest"]["return_error"] for r in backtested) / len(backtested), 4
        )
        stats["avg_predicted_return"] = round(
            sum(r["backtest"]["predicted_return"] for r in backtested) / len(backtested), 4
        )
        stats["avg_actual_return"] = round(
            sum(r["backtest"]["actual_return"] for r in backtested) / len(backtested), 4
        )

        # Group by recommendation
        stats["by_recommendation"] = self._group_stats(
            backtested, key_fn=lambda r: r["recommendation"]
        )

        # Group by industry
        stats["by_industry"] = self._group_stats(
            backtested, key_fn=lambda r: r["industry"]
        )

        # Group by score range
        def score_bucket(r):
            s = r["score"]
            if s < 40:
                return "0-40"
            elif s < 65:
                return "40-65"
            elif s < 80:
                return "65-80"
            else:
                return "80-100"

        stats["by_score_range"] = self._group_stats(backtested, key_fn=score_bucket)

        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_max_drawdown(prices: List[float]) -> float:
        """Calculate maximum peak-to-trough drawdown as a percentage."""
        if len(prices) < 2:
            return 0.0
        peak = prices[0]
        max_dd = 0.0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _group_stats(records: List[dict], key_fn) -> dict:
        """Group backtested records by a key function and compute per-group stats."""
        groups: dict = {}
        for r in records:
            key = key_fn(r)
            groups.setdefault(key, []).append(r)

        result = {}
        for key, group in groups.items():
            direction_hits = sum(1 for r in group if r["backtest"]["direction_correct"])
            result[key] = {
                "count": len(group),
                "avg_actual_return": round(
                    sum(r["backtest"]["actual_return"] for r in group) / len(group), 4
                ),
                "direction_accuracy": round(direction_hits / len(group), 4),
            }
        return result

    def _update_record(self, updated_record: dict) -> None:
        """Rewrite PREDICTIONS_FILE with the updated record in place."""
        records = self.get_all_records()
        target_id = updated_record["id"]
        new_lines = []
        for r in records:
            if r["id"] == target_id:
                new_lines.append(json.dumps(updated_record, ensure_ascii=False))
            else:
                new_lines.append(json.dumps(r, ensure_ascii=False))

        with open(config.PREDICTIONS_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        logger.debug("Updated record %s in predictions file", target_id)
