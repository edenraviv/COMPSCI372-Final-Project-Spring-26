import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan
    

def flatten(raw: dict) -> pd.DataFrame:
    '''One row per candle. Extracts all price/volume/bid/ask fields.

    series_id strips the last "-suffix" off the market_id so all mutually
    exclusive options under the same event share a group key. Used by the
    splitters to keep an entire event in a single partition.'''
    rows = []
    for market_id, candles in raw.items():
        series_id = market_id.rsplit("-", 1)[0]
        for c in candles:
            row = {
                "market_id":     market_id,
                "series_id":     series_id,
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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    '''Two documented data quality challenges:
    Challenge 1 — Missing values
    Many early candles have no price_mean, bid_close, or ask_close because
    the market has no trades yet. We flag these with indicator columns before
    filling with sentinel (-999 at scaling time) so the model can learn that
    missingness itself is informative (thin/new market).

    Challenge 2 — Volume/OI outliers
    Resolution-hour volume spikes can be 100x the typical hourly volume.
    Left uncapped these dominate tree splits. We clip at the per-market 99th
    percentile, computed on training data only to prevent leakage.'''
    
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

def drop_resolution_candle(df: pd.DataFrame) -> pd.DataFrame:
    '''The final candle of each market is the resolution candle — price has already
    collapsed to $0.01 (NO) or risen to $1.00 (YES). This candle is never
    available at inference time on a live market, so dropping it keeps training
    and inference consistent and prevents label leakage via the final price.'''
    
    mask = df.groupby("market_id").cumcount(ascending=False) > 0
    dropped = (~mask).sum()
    print(f"Dropped {dropped} resolution candles (1 per market)")
    return df[mask].reset_index(drop=True)


def three_way_split(df: pd.DataFrame, seed: int = 42):
    '''TRAIN / VAL / TEST SPLIT — 70 / 15 / 15
    GroupShuffleSplit groups by series_id so all mutually exclusive option
    markets under the same event stay in one partition. Splitting by
    market_id alone leaks: e.g. if KXTRUMPMENTION-26FEB20-A is in train and
    KXTRUMPMENTION-26FEB20-B is in val, the model can learn series-specific
    price dynamics and complement structure (one option YES => others NO)
    from train and exploit it on val.'''

    groups = df["series_id"].values

    spl1 = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO,
                             random_state=seed)
    tv_idx, test_idx = next(spl1.split(df, groups=groups))
    df_tv   = df.iloc[tv_idx]
    df_test = df.iloc[test_idx]

    val_of_tv = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    spl2 = GroupShuffleSplit(n_splits=1, test_size=val_of_tv,
                             random_state=seed)
    tr_idx, val_idx = next(spl2.split(df_tv,
                                      groups=df_tv["series_id"].values))
    df_train = df_tv.iloc[tr_idx]
    df_val   = df_tv.iloc[val_idx]

    n = len(df)
    print(f"\nSplit — Train {TRAIN_RATIO:.0%} / Val {VAL_RATIO:.0%}"
          f" / Test {TEST_RATIO:.0%}")
    for name, part in [("Train", df_train), ("Val", df_val),
                        ("Test",  df_test)]:
        print(f"  {name:<5}: {len(part):>5} rows | "
              f"{part['market_id'].nunique():>3} markets | "
              f"{part['series_id'].nunique():>3} series | "
              f"{len(part)/n*100:.1f}%")
    return df_train, df_val, df_test