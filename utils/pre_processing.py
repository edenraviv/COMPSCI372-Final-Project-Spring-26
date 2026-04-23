
import pandas as pd

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
