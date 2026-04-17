import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model
MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = 512
DEVICE = None  # auto-detect: cuda > mps > cpu

# Prediction
PRED_LEN = 40
SAMPLE_COUNT = 5
TEMPERATURE = 1.0
TOP_P = 0.9
LOOKBACK = 400

# Scoring weights (6 dimensions, sum to 1.0)
SCORE_WEIGHTS = {
    "expected_return": 0.30,
    "max_drawdown": 0.20,
    "uncertainty": 0.15,
    "technical": 0.15,
    "fund_flow": 0.10,
    "margin": 0.10,
}

# Scanner
SCAN_INDICES = ["hs300", "zz500"]
SCAN_BATCH_SIZE = 16
SCAN_TOP_N = 20
SCAN_CACHE_HOURS = 4

# A-share price limit
PRICE_LIMIT_RATE = 0.10

# Cache
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
WATCHLIST_FILE = os.path.join(BASE_DIR, "data", "watchlist.json")
PREDICTIONS_FILE = os.path.join(BASE_DIR, "data", "predictions.jsonl")
OPTIMIZED_WEIGHTS_FILE = os.path.join(BASE_DIR, "data", "optimized_weights.json")

# Finetune
FINETUNE_DATA_DIR = os.path.join(BASE_DIR, "data", "finetune")
FINETUNE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "finetune_output")

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
