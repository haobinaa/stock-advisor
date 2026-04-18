import sys
import os
import logging
import threading

import numpy as np
import pandas as pd

from app import config

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so `model` package is importable
_project_root = os.path.dirname(config.BASE_DIR)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class StockPredictor:
    """Singleton wrapper around KronosPredictor for stock price prediction."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._predictor = None
        self._load_model()

    def _load_model(self):
        try:
            from model import KronosTokenizer, Kronos, KronosPredictor

            logger.info(
                "Loading Kronos model=%s tokenizer=%s device=%s",
                config.MODEL_NAME,
                config.TOKENIZER_NAME,
                config.DEVICE or "auto",
            )
            tokenizer = KronosTokenizer.from_pretrained(config.TOKENIZER_NAME)
            model = Kronos.from_pretrained(config.MODEL_NAME)
            self._predictor = KronosPredictor(
                model,
                tokenizer,
                device=config.DEVICE,
                max_context=config.MAX_CONTEXT,
            )
            logger.info("Kronos model loaded successfully on %s", self._predictor.device)
        except Exception:
            logger.exception("Failed to load Kronos model")
            raise

    @classmethod
    def get_instance(cls) -> "StockPredictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def reload_model(self, model_path: str = None, tokenizer_path: str = None):
        """Reload model from a local fine-tuned path or default config."""
        with self._lock:
            if model_path:
                logger.info("Reloading model from local path: %s", model_path)
            self._load_model_from(model_path, tokenizer_path)

    def _load_model_from(self, model_path=None, tokenizer_path=None):
        """Load model/tokenizer from specific paths or defaults."""
        try:
            from model import KronosTokenizer, Kronos, KronosPredictor

            tok_src = tokenizer_path or config.TOKENIZER_NAME
            mod_src = model_path or config.MODEL_NAME
            logger.info("Loading tokenizer=%s model=%s", tok_src, mod_src)

            tokenizer = KronosTokenizer.from_pretrained(tok_src)
            model = Kronos.from_pretrained(mod_src)
            self._predictor = KronosPredictor(
                model, tokenizer, device=config.DEVICE, max_context=config.MAX_CONTEXT,
            )
            logger.info("Model reloaded on %s", self._predictor.device)
        except Exception:
            logger.exception("Failed to reload model")
            raise

    # ------------------------------------------------------------------
    # Price limit clipping (A-share ±10%)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_price_limit(pred_df: pd.DataFrame, last_close: float) -> pd.DataFrame:
        """Clip OHLC predictions to A-share daily price limits row by row."""
        price_cols = ["open", "high", "low", "close"]
        rate = config.PRICE_LIMIT_RATE
        df = pred_df.copy()
        prev_close = last_close
        for idx in df.index:
            upper = prev_close * (1 + rate)
            lower = prev_close * (1 - rate)
            for col in price_cols:
                df.at[idx, col] = np.clip(df.at[idx, col], lower, upper)
            prev_close = df.at[idx, "close"]
        return df

    # ------------------------------------------------------------------
    # Single stock prediction
    # ------------------------------------------------------------------

    def predict_single(
        self,
        df: pd.DataFrame,
        pred_len: int = config.PRED_LEN,
        sample_count: int = config.SAMPLE_COUNT,
        apply_price_limit: bool = True,
    ) -> dict:
        """Predict future prices for a single stock.

        Args:
            df: Historical data with columns [open, high, low, close, volume, amount]
                and either a 'date' column or a DatetimeIndex.
            pred_len: Number of future trading days to predict.
            sample_count: Number of independent sample paths to generate.

        Returns:
            dict with keys:
                predictions - list of DataFrames (one per sample path)
                mean_pred   - DataFrame of mean predictions
                std_pred    - DataFrame of prediction std
        """
        df = df.copy()

        # Ensure input does not exceed max_context to prevent tensor size mismatch
        if len(df) > config.MAX_CONTEXT:
            df = df.tail(config.MAX_CONTEXT).reset_index(drop=True)

        # Extract date info and prepare timestamp series
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = pd.Series(df.index)
            df = df.reset_index(drop=True)
        else:
            # No date info - create synthetic business day range ending today
            dates = pd.Series(pd.bdate_range(end=pd.Timestamp.now(), periods=len(df)))

        x_timestamp = dates.reset_index(drop=True)
        last_date = x_timestamp.iloc[-1]
        y_timestamp = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len))

        last_close = float(df["close"].iloc[-1])

        predictions: list[pd.DataFrame] = []
        for _ in range(sample_count):
            pred_df = self._predictor.predict(
                df,
                x_timestamp,
                y_timestamp,
                pred_len,
                T=config.TEMPERATURE,
                top_p=config.TOP_P,
                sample_count=1,
            )
            pred_df = self._apply_price_limit(pred_df, last_close) if apply_price_limit else pred_df
            predictions.append(pred_df)

        all_values = np.stack([p.values for p in predictions], axis=0)
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        cols = predictions[0].columns
        idx = predictions[0].index
        mean_pred = pd.DataFrame(mean_values, columns=cols, index=idx)
        std_pred = pd.DataFrame(std_values, columns=cols, index=idx)

        return {
            "predictions": predictions,
            "mean_pred": mean_pred,
            "std_pred": std_pred,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        df_list: list[pd.DataFrame],
        pred_len: int = config.PRED_LEN,
        sample_count: int = config.SAMPLE_COUNT,
    ) -> list[dict]:
        """Batch predict for multiple stocks.

        Uses the model's native batch inference when all series share the
        same historical length; otherwise falls back to sequential calls.

        Args:
            df_list: List of historical DataFrames (same format as predict_single).
            pred_len: Number of future trading days to predict.
            sample_count: Number of independent sample paths per stock.

        Returns:
            List of dicts (same format as predict_single output).
        """
        prepared: list[tuple[pd.DataFrame, pd.Series, pd.Series, float]] = []
        for df in df_list:
            df = df.copy()
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])
            elif isinstance(df.index, pd.DatetimeIndex):
                dates = pd.Series(df.index)
                df = df.reset_index(drop=True)
            else:
                dates = pd.Series(pd.bdate_range(end=pd.Timestamp.now(), periods=len(df)))

            x_ts = dates.reset_index(drop=True)
            last_date = x_ts.iloc[-1]
            y_ts = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len))
            last_close = float(df["close"].iloc[-1])
            prepared.append((df, x_ts, y_ts, last_close))

        # Check whether native batch is feasible (equal historical lengths)
        lengths = [len(t[0]) for t in prepared]
        can_batch = len(set(lengths)) == 1

        if can_batch:
            return self._batch_native(prepared, pred_len, sample_count)
        else:
            logger.info(
                "Series lengths differ (%s), falling back to sequential prediction",
                lengths,
            )
            return self._batch_sequential(prepared, pred_len, sample_count)

    def _batch_native(
        self,
        prepared: list[tuple],
        pred_len: int,
        sample_count: int,
    ) -> list[dict]:
        df_list = [p[0] for p in prepared]
        x_ts_list = [p[1] for p in prepared]
        y_ts_list = [p[2] for p in prepared]
        last_closes = [p[3] for p in prepared]

        all_predictions: list[list[pd.DataFrame]] = [[] for _ in range(len(df_list))]

        for _ in range(sample_count):
            pred_dfs = self._predictor.predict_batch(
                df_list,
                x_ts_list,
                y_ts_list,
                pred_len,
                T=config.TEMPERATURE,
                top_p=config.TOP_P,
                sample_count=1,
            )
            for i, pred_df in enumerate(pred_dfs):
                pred_df = self._apply_price_limit(pred_df, last_closes[i])
                all_predictions[i].append(pred_df)

        results = []
        for preds in all_predictions:
            stacked = np.stack([p.values for p in preds], axis=0)
            mean_vals = np.mean(stacked, axis=0)
            std_vals = np.std(stacked, axis=0)
            cols = preds[0].columns
            idx = preds[0].index
            results.append({
                "predictions": preds,
                "mean_pred": pd.DataFrame(mean_vals, columns=cols, index=idx),
                "std_pred": pd.DataFrame(std_vals, columns=cols, index=idx),
            })
        return results

    def _batch_sequential(
        self,
        prepared: list[tuple],
        pred_len: int,
        sample_count: int,
    ) -> list[dict]:
        results = []
        for df, x_ts, y_ts, last_close in prepared:
            predictions: list[pd.DataFrame] = []
            for _ in range(sample_count):
                pred_df = self._predictor.predict(
                    df,
                    x_ts,
                    y_ts,
                    pred_len,
                    T=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    sample_count=1,
                )
                pred_df = self._apply_price_limit(pred_df, last_close)
                predictions.append(pred_df)

            stacked = np.stack([p.values for p in predictions], axis=0)
            mean_vals = np.mean(stacked, axis=0)
            std_vals = np.std(stacked, axis=0)
            cols = predictions[0].columns
            idx = predictions[0].index
            results.append({
                "predictions": predictions,
                "mean_pred": pd.DataFrame(mean_vals, columns=cols, index=idx),
                "std_pred": pd.DataFrame(std_vals, columns=cols, index=idx),
            })
        return results
