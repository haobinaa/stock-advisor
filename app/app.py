import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS

# Add project root to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Lazy-loaded singletons ---
_fetcher = None
_predictor = None
_tech = None
_analyzer = None
_scanner = None
_backtest = None
_weight_optimizer = None
_pred_filter = None
_finetune_manager = None


def get_fetcher():
    global _fetcher
    if _fetcher is None:
        from app.services.data_fetcher import DataFetcher
        _fetcher = DataFetcher()
    return _fetcher


def get_predictor():
    global _predictor
    if _predictor is None:
        from app.services.predictor import StockPredictor
        _predictor = StockPredictor.get_instance()
    return _predictor


def get_tech():
    global _tech
    if _tech is None:
        from app.services.technical import TechnicalAnalyzer
        _tech = TechnicalAnalyzer()
    return _tech


def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from app.services.analyzer import StockAnalyzer
        _analyzer = StockAnalyzer()
    return _analyzer


def get_backtest():
    global _backtest
    if _backtest is None:
        from app.services.backtest import PredictionLogger
        _backtest = PredictionLogger()
    return _backtest


def get_pred_filter():
    global _pred_filter
    if _pred_filter is None:
        from app.services.prediction_filter import PredictionFilter
        stats = get_backtest().get_stats()
        _pred_filter = PredictionFilter(backtest_stats=stats)
    return _pred_filter


def get_weight_optimizer():
    global _weight_optimizer
    if _weight_optimizer is None:
        from app.services.weight_optimizer import WeightOptimizer
        _weight_optimizer = WeightOptimizer()
    return _weight_optimizer


def get_scanner():
    global _scanner
    if _scanner is None:
        from app.services.scanner import StockScanner
        _scanner = StockScanner(
            get_fetcher(), get_predictor(), get_tech(), get_analyzer(),
            prediction_filter=get_pred_filter(),
        )
    return _scanner


def get_finetune_manager():
    global _finetune_manager
    if _finetune_manager is None:
        from app.services.finetune_manager import FinetuneManager
        _finetune_manager = FinetuneManager(
            data_fetcher=get_fetcher(), backtest_logger=get_backtest(),
        )
    return _finetune_manager


# --- Watchlist helpers ---

def _load_watchlist():
    try:
        with open(config.WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_watchlist(items):
    os.makedirs(os.path.dirname(config.WATCHLIST_FILE), exist_ok=True)
    with open(config.WATCHLIST_FILE, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# --- Helper to serialize AnalysisResult ---

def _result_to_dict(r):
    d = asdict(r)
    return d


# --- Page routes ---

@app.route("/")
def index():
    return redirect(url_for("watchlist_page"))


@app.route("/watchlist")
def watchlist_page():
    return render_template("watchlist.html")


@app.route("/discover")
def discover_page():
    return render_template("discover.html")


@app.route("/detail/<symbol>")
def detail_page(symbol):
    return render_template("detail.html", symbol=symbol)


@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html")


@app.route("/finetune")
def finetune_page():
    return render_template("finetune.html")


# --- API routes ---

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        results = get_fetcher().search_stock(q)
        return jsonify(results or [])
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist", methods=["GET"])
def api_watchlist_get():
    return jsonify(_load_watchlist())


@app.route("/api/watchlist", methods=["POST"])
def api_watchlist_add():
    data = request.get_json(force=True)
    symbol = data.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    items = _load_watchlist()
    if any(item["symbol"] == symbol for item in items):
        return jsonify({"error": "already in watchlist"}), 409

    # Prefer name from request, fallback to API lookup
    name = data.get("name", "").strip()
    if not name or name == symbol:
        info = get_fetcher().get_stock_info(symbol)
        name = info.get("name", symbol) if info else symbol

    market = "a_share"
    if symbol.startswith("HK") or len(symbol) == 5:
        market = "hk"

    item = {
        "symbol": symbol,
        "name": name,
        "market": market,
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    items.append(item)
    _save_watchlist(items)
    return jsonify(item), 201


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def api_watchlist_delete(symbol):
    items = _load_watchlist()
    items = [i for i in items if i["symbol"] != symbol]
    _save_watchlist(items)
    return jsonify({"ok": True})


@app.route("/api/analyze/<symbol>")
def api_analyze(symbol):
    try:
        fetcher = get_fetcher()

        # Get history
        df = fetcher.get_stock_history(symbol, lookback=config.LOOKBACK)
        if df is None:
            return jsonify({"error": f"数据获取失败，请检查网络后重试"}), 502
        if len(df) < 100:
            return jsonify({"error": f"历史数据不足（需要100条，仅获取{len(df)}条）"}), 400

        # Basic info
        info = fetcher.get_stock_info(symbol)
        name = info.get("name", symbol) if info else symbol
        industry = info.get("industry", "") if info else ""

        current_price = float(df["close"].iloc[-1])
        change_pct = 0.0
        if len(df) >= 2:
            prev = float(df["close"].iloc[-2])
            if prev > 0:
                change_pct = round((current_price / prev - 1) * 100, 2)

        # Kronos prediction
        predictor = get_predictor()
        pred_result = predictor.predict_single(
            df[["open", "high", "low", "close", "volume", "amount"]],
            pred_len=config.PRED_LEN,
            sample_count=config.SAMPLE_COUNT,
        )

        # Technical analysis
        tech_result = get_tech().analyze(df)

        # Fund flow
        fund_flow = None
        try:
            ff_df = fetcher.get_fund_flow(symbol, days=5)
            if ff_df is not None and not ff_df.empty:
                from app.models.stock import FundFlowResult
                total_inflow = float(ff_df["main_net_inflow"].sum()) if "main_net_inflow" in ff_df.columns else 0
                # Score: normalize around 0, positive = good
                ff_score = min(100, max(0, 50 + total_inflow / 1e8 * 10))
                fund_flow = FundFlowResult(
                    main_net_inflow_5d=total_inflow,
                    fund_flow_score=ff_score,
                )
        except Exception as e:
            logger.warning(f"Fund flow fetch failed for {symbol}: {e}")

        # Margin data
        margin = None
        try:
            mg_df = fetcher.get_margin_data(symbol, days=20)
            if mg_df is not None and not mg_df.empty and "margin_balance" in mg_df.columns:
                from app.models.stock import MarginResult
                balances = mg_df["margin_balance"].dropna()
                if len(balances) >= 2:
                    trend = (float(balances.iloc[-1]) / float(balances.iloc[0]) - 1) * 100
                    mg_score = min(100, max(0, 50 + trend * 5))
                    margin = MarginResult(
                        margin_balance_trend=round(trend, 2),
                        margin_score=mg_score,
                    )
        except Exception as e:
            logger.warning(f"Margin data fetch failed for {symbol}: {e}")

        # Composite analysis
        result = get_analyzer().analyze(
            symbol=symbol,
            name=name,
            industry=industry,
            current_price=current_price,
            change_pct=change_pct,
            mean_pred=pred_result["mean_pred"],
            std_pred=pred_result["std_pred"],
            technical=tech_result,
            fund_flow=fund_flow,
            margin=margin,
        )

        # Historical prices for chart
        hist_prices = df[["close"]].tail(60).values.flatten().tolist()
        hist_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                      for d in df["date"].tail(60)]

        # Log prediction for backtest tracking
        try:
            analyzer = get_analyzer()
            dim_scores = analyzer.get_last_dim_scores()
            get_backtest().log_prediction(result, dim_scores=dim_scores)
        except Exception as e:
            logger.warning(f"Failed to log prediction for {symbol}: {e}")

        resp = _result_to_dict(result)
        resp["hist_prices"] = hist_prices
        resp["hist_dates"] = hist_dates

        return jsonify(resp)

    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}", exc_info=True)
        return jsonify({"error": f"分析异常: {e}"}), 500


@app.route("/api/analyze/batch", methods=["POST"])
def api_analyze_batch():
    data = request.get_json(force=True)
    symbols = data.get("symbols", [])
    if not symbols:
        return jsonify([])

    results = []
    for sym in symbols:
        try:
            # Reuse single analysis endpoint logic
            with app.test_request_context(f"/api/analyze/{sym}"):
                resp = api_analyze(sym)
                if resp.status_code == 200:
                    results.append(resp.get_json())
        except Exception as e:
            logger.warning(f"Batch analysis failed for {sym}: {e}")

    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return jsonify(results)


@app.route("/api/scan/start", methods=["POST"])
def api_scan_start():
    scanner = get_scanner()
    cached = scanner.get_cached_results()
    if cached:
        return jsonify({"status": "cached", "count": len(cached)})

    scanner.scan_async()
    return jsonify({"status": "started"})


@app.route("/api/scan/progress")
def api_scan_progress():
    return jsonify(get_scanner().get_scan_progress())


@app.route("/api/scan/results")
def api_scan_results():
    scanner = get_scanner()
    results = scanner.get_cached_results()
    if results is None:
        progress = scanner.get_scan_progress()
        if progress["status"] == "completed":
            results = scanner._results
        else:
            return jsonify({"status": progress["status"], "results": []})

    return jsonify({
        "status": "completed",
        "results": [_result_to_dict(r) for r in results],
    })


# --- Backtest API ---

@app.route("/api/backtest/stats")
def api_backtest_stats():
    return jsonify(get_backtest().get_stats())


@app.route("/api/backtest/records")
def api_backtest_records():
    records = get_backtest().get_all_records()
    # Most recent first
    records.sort(key=lambda r: r.get("predicted_at", ""), reverse=True)
    limit = request.args.get("limit", 50, type=int)
    return jsonify(records[:limit])


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    try:
        count = get_backtest().run_all_pending(get_fetcher())
        return jsonify({"backtested": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Weight optimization API ---

@app.route("/api/weights/current")
def api_weights_current():
    return jsonify({
        "weights": get_analyzer().current_weights,
        "is_optimized": os.path.exists(config.OPTIMIZED_WEIGHTS_FILE),
    })


@app.route("/api/weights/optimize", methods=["POST"])
def api_weights_optimize():
    try:
        optimizer = get_weight_optimizer()
        records = get_backtest().get_all_records()
        new_weights = optimizer.optimize(records)
        if new_weights is None:
            return jsonify({"error": "Not enough backtested records (need 20+)"}), 400

        old_weights = get_analyzer().current_weights
        evaluation = optimizer.evaluate(old_weights, new_weights, records)
        optimizer.save_weights(new_weights)
        get_analyzer().reload_weights()

        # Reset filter with new stats
        global _pred_filter
        _pred_filter = None

        return jsonify({
            "old_weights": old_weights,
            "new_weights": new_weights,
            "evaluation": evaluation,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Finetune API ---

@app.route("/api/finetune/status")
def api_finetune_status():
    return jsonify(get_finetune_manager().get_training_stats())


@app.route("/api/finetune/prepare", methods=["POST"])
def api_finetune_prepare():
    try:
        stats = get_finetune_manager().prepare_training_data()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/finetune/start", methods=["POST"])
def api_finetune_start():
    result = get_finetune_manager().start_finetune_async()
    return jsonify(result)


@app.route("/api/finetune/switch", methods=["POST"])
def api_finetune_switch():
    try:
        mgr = get_finetune_manager()
        model_path = mgr.get_finetuned_model_path()
        tok_path = mgr.get_finetuned_tokenizer_path()
        if not model_path:
            return jsonify({"error": "No fine-tuned model found"}), 404
        get_predictor().reload_model(model_path, tok_path)
        return jsonify({"status": "switched", "model_path": model_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Main ---

if __name__ == "__main__":
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    logger.info(f"Starting Kronos Stock Advisor on port {config.FLASK_PORT}")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
