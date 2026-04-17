import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

from app import config

logger = logging.getLogger(__name__)

_project_root = os.path.dirname(config.BASE_DIR)


class FinetuneManager:
    """Manages model fine-tuning pipeline using finetune_csv/ scripts."""

    def __init__(self, data_fetcher=None, backtest_logger=None):
        self.fetcher = data_fetcher
        self.backtest = backtest_logger
        self._status = {"state": "idle", "message": "", "progress": 0}
        self._lock = threading.Lock()

        os.makedirs(config.FINETUNE_DATA_DIR, exist_ok=True)
        os.makedirs(config.FINETUNE_OUTPUT_DIR, exist_ok=True)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _set_status(self, state: str, message: str, progress: int = 0):
        with self._lock:
            self._status = {"state": state, "message": message, "progress": progress}

    # ------------------------------------------------------------------
    # Step 1: Prepare training data
    # ------------------------------------------------------------------

    def prepare_training_data(
        self, symbols: Optional[list] = None, months: int = 6
    ) -> dict:
        """Collect recent A-share data and hard examples into a CSV for fine-tuning.

        Returns dict with stats about the prepared data.
        """
        self._set_status("preparing", "Collecting training data...")

        if self.fetcher is None:
            from app.services.data_fetcher import DataFetcher
            self.fetcher = DataFetcher()

        # Determine symbols
        if not symbols:
            try:
                symbols = self.fetcher.get_index_components("hs300")[:100]
            except Exception:
                symbols = []

        if not symbols:
            self._set_status("error", "No symbols available")
            return {"error": "No symbols"}

        all_rows = []
        for i, sym in enumerate(symbols):
            try:
                df = self.fetcher.get_stock_history(sym, lookback=int(months * 22))
                if df is not None and len(df) >= 60:
                    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                    df.insert(0, "symbol", sym)
                    all_rows.append(df)
            except Exception as e:
                logger.warning("Skip %s: %s", sym, e)

            if (i + 1) % 20 == 0:
                self._set_status("preparing", f"Fetched {i+1}/{len(symbols)}", int((i+1)/len(symbols)*50))

        if not all_rows:
            self._set_status("error", "No data collected")
            return {"error": "No data"}

        combined = pd.concat(all_rows, ignore_index=True)

        # Add hard examples from backtest (predictions with largest errors)
        hard_count = 0
        if self.backtest:
            try:
                records = self.backtest.get_all_records()
                backtested = [r for r in records if r.get("backtest")]
                if backtested:
                    backtested.sort(
                        key=lambda r: abs(r["backtest"].get("return_error", 0)),
                        reverse=True,
                    )
                    hard_symbols = set()
                    for r in backtested[:int(len(backtested) * 0.2)]:
                        hard_symbols.add(r["symbol"])

                    for sym in hard_symbols:
                        df = self.fetcher.get_stock_history(sym, lookback=int(months * 22))
                        if df is not None and len(df) >= 60:
                            df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                            df.insert(0, "symbol", sym)
                            all_rows.append(df)
                            hard_count += 1

                    combined = pd.concat(all_rows, ignore_index=True)
            except Exception as e:
                logger.warning("Hard example collection failed: %s", e)

        csv_path = os.path.join(config.FINETUNE_DATA_DIR, "training_data.csv")
        combined.to_csv(csv_path, index=False)

        stats = {
            "csv_path": csv_path,
            "total_rows": len(combined),
            "symbols": len(set(combined["symbol"])),
            "hard_examples": hard_count,
            "date_range": f"{combined['date'].min()} ~ {combined['date'].max()}",
        }

        self._set_status("prepared", f"Data ready: {stats['total_rows']} rows", 50)
        logger.info("Training data prepared: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Step 2: Generate config YAML
    # ------------------------------------------------------------------

    def _generate_config(self) -> str:
        """Generate a YAML config for finetune_csv scripts."""
        csv_path = os.path.join(config.FINETUNE_DATA_DIR, "training_data.csv")
        exp_name = f"advisor_finetune_{datetime.now().strftime('%Y%m%d')}"

        cfg = {
            "data": {
                "data_path": csv_path,
                "lookback_window": config.LOOKBACK,
                "predict_window": config.PRED_LEN,
                "max_context": config.MAX_CONTEXT,
                "clip": 5.0,
                "train_ratio": 0.9,
                "val_ratio": 0.1,
                "test_ratio": 0.0,
            },
            "training": {
                "tokenizer_epochs": 10,
                "basemodel_epochs": 5,
                "batch_size": 16,
                "log_interval": 50,
                "num_workers": 4,
                "seed": 42,
                "tokenizer_learning_rate": 0.0002,
                "predictor_learning_rate": 0.000001,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_weight_decay": 0.1,
                "accumulation_steps": 1,
            },
            "model_paths": {
                "pretrained_tokenizer": config.TOKENIZER_NAME,
                "pretrained_predictor": config.MODEL_NAME,
                "exp_name": exp_name,
                "base_path": config.FINETUNE_OUTPUT_DIR,
                "base_save_path": "",
                "finetuned_tokenizer": "",
                "tokenizer_save_name": "tokenizer",
                "basemodel_save_name": "basemodel",
            },
            "experiment": {
                "name": exp_name,
                "description": "Auto finetune from Stock Advisor",
                "use_comet": False,
                "train_tokenizer": True,
                "train_basemodel": True,
                "skip_existing": False,
            },
            "device": {
                "use_cuda": True,
                "device_id": 0,
            },
        }

        config_path = os.path.join(config.FINETUNE_DATA_DIR, "finetune_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        return config_path

    # ------------------------------------------------------------------
    # Step 3: Run fine-tuning
    # ------------------------------------------------------------------

    def start_finetune_async(self):
        """Start fine-tuning in a background thread."""
        if self._status.get("state") == "running":
            return {"error": "Already running"}

        t = threading.Thread(target=self._run_finetune, daemon=True)
        t.start()
        return {"status": "started"}

    def _run_finetune(self):
        self._set_status("running", "Generating config...", 55)

        try:
            config_path = self._generate_config()
            script_dir = os.path.join(_project_root, "finetune_csv")
            script = os.path.join(script_dir, "train_sequential.py")

            if not os.path.exists(script):
                self._set_status("error", "finetune_csv/train_sequential.py not found")
                return

            self._set_status("running", "Training in progress...", 60)
            logger.info("Starting finetune: %s --config %s", script, config_path)

            python = sys.executable
            result = subprocess.run(
                [python, script, "--config", config_path],
                cwd=script_dir,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode == 0:
                self._set_status("completed", "Fine-tuning completed", 100)
                logger.info("Finetune completed successfully")
            else:
                err_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                self._set_status("error", f"Training failed: {err_msg}", 0)
                logger.error("Finetune failed: %s", err_msg)

        except subprocess.TimeoutExpired:
            self._set_status("error", "Training timed out (2h limit)")
        except Exception as e:
            self._set_status("error", str(e))
            logger.exception("Finetune error")

    # ------------------------------------------------------------------
    # Step 4: Evaluate and switch model
    # ------------------------------------------------------------------

    def get_finetuned_model_path(self) -> Optional[str]:
        """Find the latest fine-tuned model path."""
        if not os.path.exists(config.FINETUNE_OUTPUT_DIR):
            return None

        for entry in sorted(os.listdir(config.FINETUNE_OUTPUT_DIR), reverse=True):
            model_dir = os.path.join(config.FINETUNE_OUTPUT_DIR, entry, "basemodel", "best_model")
            if os.path.exists(model_dir):
                return model_dir

        return None

    def get_finetuned_tokenizer_path(self) -> Optional[str]:
        """Find the latest fine-tuned tokenizer path."""
        if not os.path.exists(config.FINETUNE_OUTPUT_DIR):
            return None

        for entry in sorted(os.listdir(config.FINETUNE_OUTPUT_DIR), reverse=True):
            tok_dir = os.path.join(config.FINETUNE_OUTPUT_DIR, entry, "tokenizer", "best_model")
            if os.path.exists(tok_dir):
                return tok_dir

        return None

    def get_training_stats(self) -> dict:
        """Get info about training data and finetune status."""
        csv_path = os.path.join(config.FINETUNE_DATA_DIR, "training_data.csv")
        stats = {
            "data_ready": os.path.exists(csv_path),
            "finetuned_model": self.get_finetuned_model_path(),
            "finetuned_tokenizer": self.get_finetuned_tokenizer_path(),
            "status": self.get_status(),
        }

        if stats["data_ready"]:
            try:
                df = pd.read_csv(csv_path, nrows=0)
                row_count = sum(1 for _ in open(csv_path)) - 1
                stats["data_rows"] = row_count
                stats["data_columns"] = list(df.columns)
            except Exception:
                pass

        return stats
