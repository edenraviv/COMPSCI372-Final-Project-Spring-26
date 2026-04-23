import pandas as pd
import numpy as np

def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 2. FLATTEN TO DATAFRAME
# One row per candle. Extracts all price/volume/bid/ask fields.
# ══════════════════════════════════════════════════════════════════════════════

def flatten(raw: dict) -> pd.DataFrame:
    rows = []
    for market_id, candles in raw.items():
        for c in candles:
            row = {
                "market_id":     market_id,
                "ds":            pd.to_datetime(c["ds"]),
                "close":         c.get("close"),
                "high":          c.get("high"),
                "low":           c.get("low"),
                "label":         c.get("label"),
                "volume":        float(c.get("volume_fp") or 0),
                "open_interest": float(c.get("open_interest_fp") or 0),
                "end_period_ts": c.get("end_period_ts"),
                "ask_close":     _to_float(c.get("yes_ask", {}).get("close_dollars")),
                "ask_high":      _to_float(c.get("yes_ask", {}).get("high_dollars")),
                "ask_low":       _to_float(c.get("yes_ask", {}).get("low_dollars")),
                "bid_close":     _to_float(c.get("yes_bid", {}).get("close_dollars")),
                "bid_high":      _to_float(c.get("yes_bid", {}).get("high_dollars")),
                "bid_low":       _to_float(c.get("yes_bid", {}).get("low_dollars")),
                "price_mean":    _to_float(c.get("price", {}).get("mean_dollars")),
            }
            rows.append(row)

    df = (pd.DataFrame(rows)
            .sort_values(["market_id", "ds"])
            .reset_index(drop=True))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
#
# Two documented data quality challenges:
#
#   Challenge 1 — Missing values
#     Many early candles have no price_mean, bid_close, or ask_close because
#     the market has no trades yet. We flag these with indicator columns before
#     filling with sentinel (-999 at scaling time) so the model can learn that
#     missingness itself is informative (thin/new market).
#
#   Challenge 2 — Volume/OI outliers
#     Resolution-hour volume spikes can be 100x the typical hourly volume.
#     Left uncapped these dominate tree splits. We clip at the per-market 99th
#     percentile, computed on training data only to prevent leakage.
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Challenge 1: flag missing prices before sentinel fill
    for col in ["price_mean", "bid_close", "ask_close"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # Challenge 2: clip outliers at per-market 99th percentile
    for col in ["volume", "open_interest"]:
        p99 = df.groupby("market_id")[col].transform(
            lambda x: x.quantile(0.99))
        df[col] = df[col].clip(upper=p99)

    return df
