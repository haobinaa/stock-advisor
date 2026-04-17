import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from app import config
from app.models.stock import AnalysisResult

logger = logging.getLogger(__name__)


class StockScanner:

    def __init__(self, data_fetcher, predictor, technical_analyzer, analyzer,
                 prediction_filter=None):
        self.fetcher = data_fetcher
        self.predictor = predictor
        self.tech = technical_analyzer
        self.analyzer = analyzer
        self.pred_filter = prediction_filter

        self._progress = {"total": 0, "completed": 0, "status": "idle"}
        self._results: List[AnalysisResult] = []
        self._cache_time: Optional[datetime] = None
        self._lock = threading.Lock()
        self._scanning = False

    def get_scan_progress(self) -> dict:
        with self._lock:
            return dict(self._progress)

    def get_cached_results(self) -> Optional[List[AnalysisResult]]:
        if (
            self._cache_time
            and datetime.now() - self._cache_time
            < timedelta(hours=config.SCAN_CACHE_HOURS)
            and self._results
        ):
            return self._results
        return None

    def scan_async(
        self,
        index_list: Optional[List[str]] = None,
        top_n: Optional[int] = None,
    ):
        if self._scanning:
            logger.info("Scan already in progress")
            return

        cached = self.get_cached_results()
        if cached:
            logger.info("Returning cached scan results")
            return

        index_list = index_list or config.SCAN_INDICES
        top_n = top_n or config.SCAN_TOP_N

        t = threading.Thread(
            target=self._scan_worker,
            args=(index_list, top_n),
            daemon=True,
        )
        t.start()

    def _scan_worker(self, index_list: List[str], top_n: int):
        self._scanning = True
        try:
            self._do_scan(index_list, top_n)
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            with self._lock:
                self._progress["status"] = f"error: {e}"
        finally:
            self._scanning = False

    def _do_scan(self, index_list: List[str], top_n: int):
        # 1. Collect all symbols
        all_symbols = set()
        for idx in index_list:
            try:
                components = self.fetcher.get_index_components(idx)
                if components:
                    all_symbols.update(components)
            except Exception as e:
                logger.error(f"Failed to get components for {idx}: {e}")

        symbols = sorted(all_symbols)
        total = len(symbols)
        logger.info(f"Scanning {total} stocks from {index_list}")

        with self._lock:
            self._progress = {
                "total": total,
                "completed": 0,
                "status": "fetching data",
            }

        # 2. Fetch history for all symbols
        history_map = {}
        for i, sym in enumerate(symbols):
            try:
                df = self.fetcher.get_stock_history(sym, lookback=config.LOOKBACK)
                if df is not None and len(df) >= 100:
                    history_map[sym] = df
            except Exception as e:
                logger.warning(f"Skip {sym}: {e}")

            if (i + 1) % 50 == 0:
                with self._lock:
                    self._progress["completed"] = i + 1
                    self._progress["status"] = "fetching data"
                time.sleep(0.1)  # rate limit

        with self._lock:
            self._progress["status"] = "predicting"
            self._progress["completed"] = 0
            self._progress["total"] = len(history_map)

        # 3. Batch predict
        valid_symbols = list(history_map.keys())
        results: List[AnalysisResult] = []
        batch_size = config.SCAN_BATCH_SIZE
        completed = 0

        for batch_start in range(0, len(valid_symbols), batch_size):
            batch_syms = valid_symbols[batch_start : batch_start + batch_size]
            batch_dfs = [history_map[s] for s in batch_syms]

            try:
                pred_results = self.predictor.predict_batch(
                    [df[["open", "high", "low", "close", "volume", "amount"]]
                     for df in batch_dfs],
                    pred_len=config.PRED_LEN,
                    sample_count=config.SAMPLE_COUNT,
                )
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                pred_results = [None] * len(batch_syms)

            for sym, df, pred in zip(batch_syms, batch_dfs, pred_results):
                completed += 1
                if pred is None:
                    continue

                try:
                    # Get basic info
                    info = self.fetcher.get_stock_info(sym)
                    name = info.get("name", sym) if info else sym
                    industry = info.get("industry", "") if info else ""

                    # Apply prediction filter
                    if self.pred_filter:
                        excluded, reason = self.pred_filter.should_exclude(sym, name, industry)
                        if excluded:
                            logger.debug("Filtered out %s: %s", sym, reason)
                            continue

                    confidence = 1.0
                    if self.pred_filter:
                        confidence = self.pred_filter.get_confidence_adjustment(industry)

                    current_price = float(df["close"].iloc[-1])
                    change_pct = 0.0
                    if len(df) >= 2:
                        prev = float(df["close"].iloc[-2])
                        if prev > 0:
                            change_pct = (current_price / prev - 1) * 100

                    # Technical analysis
                    tech_result = self.tech.analyze(df)

                    # Fund flow (simplified for scan - skip if slow)
                    fund_flow = None
                    margin = None

                    result = self.analyzer.analyze(
                        symbol=sym,
                        name=name,
                        industry=industry,
                        current_price=current_price,
                        change_pct=change_pct,
                        mean_pred=pred["mean_pred"],
                        std_pred=pred["std_pred"],
                        technical=tech_result,
                        fund_flow=fund_flow,
                        margin=margin,
                        confidence_adjustment=confidence,
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Analysis failed for {sym}: {e}")

            with self._lock:
                self._progress["completed"] = completed
                self._progress["status"] = "predicting"

        # 4. Sort and filter
        results.sort(key=lambda r: r.score, reverse=True)
        filtered = [
            r for r in results
            if r.score >= 65 and r.risk_level in ("low", "medium")
        ]
        top_results = filtered[:top_n]

        with self._lock:
            self._results = top_results
            self._cache_time = datetime.now()
            self._progress["status"] = "completed"
            self._progress["completed"] = self._progress["total"]

        logger.info(
            f"Scan completed: {len(results)} analyzed, "
            f"{len(filtered)} passed filter, returning top {len(top_results)}"
        )
