import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from data_visualization import _plot_feature_importance


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''FEATURE ENGINEERING — 35+ derived features across 7 groups.

    All features are strictly backward-looking — only use data available at or
    before the current candle. Safe to compute on a live market mid-life.

    Intentionally excluded:
      total_hours  — requires knowing full market duration in advance
      pct_elapsed  — same reason; divides hour_index by total_hours'''

    df = df.copy()
    g  = df.groupby("market_id")

    # Microstructure
    df["bid_ask_spread"] = df["ask_close"] - df["bid_close"]
    df["midpoint"]       = (df["ask_close"] + df["bid_close"]) / 2
    df["close_vs_mid"]   = df["close"] - df["midpoint"]
    df["price_range"]    = df["high"] - df["low"]
    df["ask_range"]      = df["ask_high"] - df["ask_low"]
    df["bid_range"]      = df["bid_high"] - df["bid_low"]
    df["close_vs_mean"]  = df["close"] - df["price_mean"]

    # Momentum — shift(1) ensures strictly backward-looking
    df["close_lag1"]    = g["close"].shift(1)
    df["close_lag2"]    = g["close"].shift(2)
    df["momentum_1h"]   = df["close"] - df["close_lag1"]
    df["momentum_2h"]   = df["close"] - df["close_lag2"]
    df["ask_momentum"]  = df["ask_close"] - g["ask_close"].shift(1)
    df["bid_momentum"]  = df["bid_close"] - g["bid_close"].shift(1)
    df["spread_change"] = df["bid_ask_spread"] - g["bid_ask_spread"].shift(1)

    # Volume
    df["volume_delta"] = df["volume"] - g["volume"].shift(1)
    df["oi_delta"]     = df["open_interest"] - g["open_interest"].shift(1)
    df["vol_to_oi"]    = df["volume"] / (df["open_interest"] + 1e-6)
    df["log_volume"]   = np.log1p(df["volume"])
    df["log_oi"]       = np.log1p(df["open_interest"])

    # end_period_ts = end of this candle, not market expiry
    # So we use the last candle's ds per market as a proxy for expiry
    df["market_end_ts"] = df.groupby("market_id")["ds"].transform("max")
    df["hours_to_expiry"] = (df["market_end_ts"] - df["ds"]).dt.total_seconds() / 3600

    df["is_final_3h"]     = (df["hours_to_expiry"] <= 3).astype(int)
    df["is_final_1h"]     = (df["hours_to_expiry"] <= 1).astype(int)
    df["hour_index"]      = g.cumcount()

    # Rolling — shift(1) prevents current-candle lookahead
    for w in [2, 3]:
        df[f"close_roll_mean_{w}h"] = g["close"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"close_roll_std_{w}h"]  = g["close"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std())
        df[f"vol_roll_sum_{w}h"]    = g["volume"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).sum())

    # Level signals
    df["close_is_floor"]   = (df["close"] <= 0.01).astype(int)
    df["close_is_ceiling"] = (df["close"] >= 0.99).astype(int)
    df["first_close"]      = g["close"].transform("first")
    df["close_vs_open"]    = df["close"] - df["first_close"]
    df["running_std"]      = g["close"].transform(
        lambda x: x.expanding().std())

    # Ensure missing flag columns exist
    for col in ["price_mean_missing", "bid_close_missing", "ask_close_missing"]:
        if col not in df.columns:
            df[col] = 0

    return df


def scale_features(df_train, df_val, df_test, feature_cols):
    '''SCALE FEATURES
    StandardScaler fit on train only — same scaler applied at inference.'''
    scaler  = StandardScaler()
    X_train = df_train[feature_cols].fillna(-999).values
    X_val   = df_val[feature_cols].fillna(-999).values
    X_test  = df_test[feature_cols].fillna(-999).values
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def select_features(model, feature_cols: list,
                    X_val=None, y_val=None,
                    threshold_pct: float = 0.001):
    '''FEATURE SELECTION — importance-based pruning with documented impact.

    After the initial model is trained, features contributing less than 1% of
    total gain importance are pruned. This reduces noise, speeds up inference,
    and often improves generalization on small datasets. The impact on val
    log-loss is printed for documentation.'''
    imp   = pd.Series(model.feature_importance(importance_type="gain"),
                      index=feature_cols)
    total = imp.sum()
    pct   = imp / (total + 1e-9)

    selected = pct[pct >= threshold_pct].index.tolist()
    dropped  = pct[pct <  threshold_pct].index.tolist()

    print(f"\n── Feature Selection (threshold={threshold_pct*100:.0f}% gain) ─")
    print(f"  Kept   : {len(selected)} / {len(feature_cols)} features")
    print(f"  Dropped: {dropped}")

    # Document impact on val log-loss
    if X_val is not None and y_val is not None:
        before = log_loss(y_val, model.predict(X_val))
        print(f"  Val log-loss before pruning: {before:.4f}")
        print(f"  (Re-train on pruned set to see after)")

    _plot_feature_importance(imp)
    return selected, imp
