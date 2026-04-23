
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass
class MarketFeatures:
    """
    Ingress fetches what's needed to build this information. Add fields here as the regression head demands them.
    """
    market_id: str
    yes_price: float                            # 0-100 cents (normalize before use)
    no_price: float
    last_price_dollars: float              
    volume_history: list[float]
    price_momentum: Optional[float] = None      # e.g. 7-day delta
    volume_weighted_price: Optional[float] = None
    time_to_resolution: Optional[float] = None  # Days until resolution
    rag_query: Optional[str] = None             # Defaults to market_id if None
    label: Optional[float] = None              # Ground truth: 1.0 = YES resolved, 0.0 = NO


# ---------------------------------------------------------------------------
# Timeseries Schema
# ---------------------------------------------------------------------------

N_CANDLE_FEATURES = 16  # number of numeric features per candle (see _candle_to_row)


FEATURE_GROUPS = {
    "microstructure": [
        "bid_ask_spread", "midpoint", "close_vs_mid",
        "price_range", "ask_range", "bid_range", "close_vs_mean",
    ],
    "momentum": [
        "momentum_1h", "momentum_2h",
        "ask_momentum", "bid_momentum", "spread_change",
    ],
    "volume": [
        "volume_delta", "oi_delta", "vol_to_oi", "log_volume", "log_oi",
    ],
    "time": [
        "hours_to_expiry", "is_final_3h", "is_final_1h", "hour_index",
    ],
    "rolling": [
        "close_roll_mean_2h", "close_roll_mean_3h",
        "close_roll_std_2h",  "close_roll_std_3h",
        "vol_roll_sum_2h",    "vol_roll_sum_3h",
    ],
    "level": [
        "close_is_floor", "close_is_ceiling",
        "close_vs_open", "first_close", "running_std",
    ],
    "raw": [
        "close", "ask_close", "bid_close", "open_interest", "volume",
    ],
    "missing_flags": [
        "price_mean_missing", "bid_close_missing", "ask_close_missing",
    ],
}

ALL_FEATURE_COLS = [f for cols in FEATURE_GROUPS.values() for f in cols]
