"""
Project-wide configuration.

All tunable constants, artifact paths, and environment-sourced secrets live
here so they can be changed without editing pipeline code in src/.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ── Repo layout ──────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ── Artifact paths ───────────────────────────────────────────────────────────
DATA_SOURCE     = str(DATA_DIR / "market_timeseries.json")
MODEL_LGBM_PATH = str(ROOT_DIR / "kalshi_lgbm.txt")
MODEL_XGB_PATH  = str(ROOT_DIR / "kalshi_xgb.json")
SCALER_PATH     = str(ROOT_DIR / "kalshi_scaler.pkl")


# ── Pipeline tunables ────────────────────────────────────────────────────────
TRAIN_RATIO                 = 0.70
VAL_RATIO                   = 0.15
TEST_RATIO                  = 0.15
MIN_HOURS_TO_EXPIRY         = 6       # drop candles within N hours of expiry
BACKTEST_THRESHOLD          = 0.60    # predicted prob above which to bet YES
FEATURE_SELECTION_THRESHOLD = 0.001   # min gain fraction to keep a feature


# ── Hyperparameter search grid (LightGBM) ────────────────────────────────────
HYPERPARAM_CONFIGS = {
    "Config-A (default)": {
        "learning_rate": 0.05, "num_leaves": 20,
        "min_data_in_leaf": 10,  "lambda_l2": 1,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
    },
    "Config-B (deep+reg)": {
        "learning_rate": 0.02, "num_leaves": 50,
        "min_data_in_leaf": 10, "lambda_l2": 5.0,
        "feature_fraction": 0.7, "bagging_fraction": 0.7,
    },
    "Config-C (shallow+fast)": {
        "learning_rate": 0.10, "num_leaves": 15,
        "min_data_in_leaf": 20, "lambda_l2": 5.0,
        "feature_fraction": 0.9, "bagging_fraction": 0.9,
    },
}


# ── Kalshi API credentials (loaded from apikey.env at repo root) ─────────────
load_dotenv(ROOT_DIR / "apikey.env")
API_KEY_ID       = os.getenv("API_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH")
BASE_URL         = os.getenv("BASE_URL")
