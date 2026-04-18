#!/usr/bin/env python3
"""Historical backtest for Kronos Stock Advisor.

Simulates past predictions by slicing historical data at multiple time
points and comparing model forecasts against actual subsequent prices.

Usage:
    python scripts/historical_backtest.py
    python scripts/historical_backtest.py --symbols 600036 002594
    python scripts/historical_backtest.py --points 3 --pred-len 20
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app import config
from app.services.data_fetcher import DataFetcher
from app.services.predictor import StockPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def calc_max_drawdown(prices: list[float]) -> float:
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


def calc_mape(actual: list[float], predicted: list[float]) -> float:
    n = min(len(actual), len(predicted))
    if n == 0:
        return 0.0
    errors = []
    for i in range(n):
        if actual[i] != 0:
            errors.append(abs(actual[i] - predicted[i]) / abs(actual[i]))
    return (sum(errors) / len(errors) * 100) if errors else 0.0


def run_single_backtest(
    predictor: StockPredictor,
    full_df: pd.DataFrame,
    split_idx: int,
    pred_len: int,
    sample_count: int,
    is_hk: bool = False,
) -> dict:
    """Run one backtest: predict from split_idx, compare with actual."""
    # History: data before split point
    hist_df = full_df.iloc[:split_idx].copy()

    # Actual: data after split point
    actual_df = full_df.iloc[split_idx: split_idx + pred_len].copy()
    if len(actual_df) < 10:
        return None  # Not enough future data

    # Run prediction
    input_df = hist_df[["open", "high", "low", "close", "volume", "amount"]].copy()
    if "date" in hist_df.columns:
        input_df.insert(0, "date", hist_df["date"].values)

    pred_result = predictor.predict_single(
        input_df, pred_len=pred_len, sample_count=sample_count,
        apply_price_limit=not is_hk,
    )

    mean_pred = pred_result["mean_pred"]
    pred_closes = mean_pred["close"].tolist()

    actual_closes = actual_df["close"].tolist()
    current_price = float(hist_df["close"].iloc[-1])
    split_date = str(hist_df["date"].iloc[-1]) if "date" in hist_df.columns else f"idx_{split_idx}"

    # Align lengths
    n = min(len(pred_closes), len(actual_closes))
    pred_closes_aligned = pred_closes[:n]
    actual_closes_aligned = actual_closes[:n]

    # Metrics
    actual_return = (actual_closes_aligned[-1] / current_price - 1) * 100
    predicted_return = (pred_closes_aligned[-1] / current_price - 1) * 100
    return_error = abs(actual_return - predicted_return)
    direction_correct = (actual_return >= 0) == (predicted_return >= 0)
    mape = calc_mape(actual_closes_aligned, pred_closes_aligned)
    actual_dd = calc_max_drawdown(actual_closes_aligned)
    pred_dd = calc_max_drawdown(pred_closes_aligned)

    return {
        "split_date": split_date,
        "current_price": round(current_price, 2),
        "actual_days": n,
        "actual_return": round(actual_return, 2),
        "predicted_return": round(predicted_return, 2),
        "return_error": round(return_error, 2),
        "direction_correct": direction_correct,
        "mape": round(mape, 2),
        "actual_max_drawdown": round(actual_dd, 2),
        "predicted_max_drawdown": round(pred_dd, 2),
    }


def run_backtest_for_stock(
    symbol: str,
    name: str,
    fetcher: DataFetcher,
    predictor: StockPredictor,
    points: int,
    pred_len: int,
    sample_count: int,
) -> dict:
    """Run multiple backtest slices for one stock."""
    logger.info("=" * 60)
    logger.info("回测: %s (%s)", name, symbol)
    logger.info("=" * 60)

    # Fetch maximum history
    df = fetcher.get_stock_history(symbol, lookback=800)
    if df is None or len(df) < 200:
        logger.warning("数据不足，跳过 %s (仅 %d 条)", symbol, len(df) if df is not None else 0)
        return None

    total_rows = len(df)
    # We need at least 100 rows of history + pred_len rows of future
    min_hist = 100
    # Generate split points evenly from (min_hist) to (total_rows - pred_len)
    max_split = total_rows - pred_len
    if max_split <= min_hist:
        logger.warning("数据不足以进行回测，跳过 %s", symbol)
        return None

    split_points = np.linspace(min_hist, max_split, points, dtype=int).tolist()
    # Remove duplicates
    split_points = sorted(set(split_points))

    is_hk = DataFetcher._is_hk(symbol)

    results = []
    for i, sp in enumerate(split_points):
        logger.info("  切片 %d/%d: 使用前 %d 天数据，预测未来 %d 天", i + 1, len(split_points), sp, pred_len)
        try:
            result = run_single_backtest(predictor, df, sp, pred_len, sample_count, is_hk=is_hk)
            if result:
                results.append(result)
                logger.info(
                    "    日期=%s 预测收益=%.2f%% 实际收益=%.2f%% 方向=%s MAPE=%.2f%%",
                    result["split_date"],
                    result["predicted_return"],
                    result["actual_return"],
                    "✓" if result["direction_correct"] else "✗",
                    result["mape"],
                )
        except Exception as e:
            logger.error("  切片 %d 失败: %s", i + 1, e)

    if not results:
        return None

    # Aggregate
    direction_hits = sum(1 for r in results if r["direction_correct"])
    stock_summary = {
        "symbol": symbol,
        "name": name,
        "total_slices": len(results),
        "direction_accuracy": round(direction_hits / len(results) * 100, 1),
        "avg_return_error": round(sum(r["return_error"] for r in results) / len(results), 2),
        "avg_mape": round(sum(r["mape"] for r in results) / len(results), 2),
        "avg_predicted_return": round(sum(r["predicted_return"] for r in results) / len(results), 2),
        "avg_actual_return": round(sum(r["actual_return"] for r in results) / len(results), 2),
        "details": results,
    }
    return stock_summary


def print_summary(all_results: list[dict]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("                        历 史 回 测 报 告")
    print("=" * 80)

    # Per-stock table
    print(f"\n{'股票':<12} {'切片数':>6} {'方向准确率':>10} {'平均MAPE':>10} {'收益误差':>10} {'预测收益':>10} {'实际收益':>10}")
    print("-" * 80)

    total_slices = 0
    total_direction_hits = 0
    all_mapes = []
    all_errors = []

    for r in all_results:
        hits = int(r["direction_accuracy"] / 100 * r["total_slices"])
        total_slices += r["total_slices"]
        total_direction_hits += hits
        all_mapes.extend([d["mape"] for d in r["details"]])
        all_errors.extend([d["return_error"] for d in r["details"]])

        print(
            f"{r['name']:<12} {r['total_slices']:>6} {r['direction_accuracy']:>9.1f}% "
            f"{r['avg_mape']:>9.2f}% {r['avg_return_error']:>9.2f}% "
            f"{r['avg_predicted_return']:>9.2f}% {r['avg_actual_return']:>9.2f}%"
        )

    print("-" * 80)

    if total_slices > 0:
        overall_dir_acc = total_direction_hits / total_slices * 100
        overall_mape = sum(all_mapes) / len(all_mapes)
        overall_error = sum(all_errors) / len(all_errors)

        print(f"{'汇总':<12} {total_slices:>6} {overall_dir_acc:>9.1f}% {overall_mape:>9.2f}% {overall_error:>9.2f}%")
        print()
        print(f"  总回测切片数:   {total_slices}")
        print(f"  方向准确率:     {overall_dir_acc:.1f}%")
        print(f"  平均MAPE:       {overall_mape:.2f}%")
        print(f"  平均收益误差:   {overall_error:.2f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Kronos Stock Advisor 历史回测")
    parser.add_argument("--symbols", nargs="*", help="指定股票代码（默认用自选股A股）")
    parser.add_argument("--points", type=int, default=5, help="每只股票的回测时间切片数")
    parser.add_argument("--pred-len", type=int, default=40, help="预测天数")
    parser.add_argument("--sample-count", type=int, default=5, help="采样路径数")
    args = parser.parse_args()

    fetcher = DataFetcher()

    # Load stocks
    if args.symbols:
        stocks = []
        for s in args.symbols:
            market = "hk" if DataFetcher._is_hk(s) else "a_share"
            stocks.append({"symbol": s, "name": s, "market": market})
    else:
        try:
            with open(config.WATCHLIST_FILE, "r") as f:
                watchlist = json.load(f)
            stocks = watchlist
        except (FileNotFoundError, json.JSONDecodeError):
            stocks = []

    if not stocks:
        print("没有可回测的股票。请添加自选股或用 --symbols 指定。")
        return

    print(f"准备回测 {len(stocks)} 只股票，每只 {args.points} 个切片，预测 {args.pred_len} 天")
    print(f"加载 Kronos 模型中...")

    predictor = StockPredictor.get_instance()
    print("模型加载完成\n")

    all_results = []
    start_time = time.time()

    for stock in stocks:
        result = run_backtest_for_stock(
            symbol=stock["symbol"],
            name=stock["name"],
            fetcher=fetcher,
            predictor=predictor,
            points=args.points,
            pred_len=args.pred_len,
            sample_count=args.sample_count,
        )
        if result:
            all_results.append(result)

    elapsed = time.time() - start_time

    if not all_results:
        print("所有股票回测失败，无结果。")
        return

    print_summary(all_results)
    print(f"\n  耗时: {elapsed:.1f} 秒")

    # Save detailed results
    output_path = os.path.join(config.BASE_DIR, "data", "historical_backtest.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "pred_len": args.pred_len,
            "sample_count": args.sample_count,
            "points": args.points,
        },
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
