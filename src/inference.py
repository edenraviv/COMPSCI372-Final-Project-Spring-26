import time
import pandas as pd
from kalshi_client import KalshiClient
from candle_pre_processing import flatten, preprocess, _to_float
from features import engineer_features
from models import load_models, ensemble_predict


def get_and_format_candles(ticker: str, series_ticker: str = None,
                   start: int = None, end: int = None,
                   period: int = 60) -> dict:
    '''KALSHI API — fetch and format live candles.

    Delegates the HTTP call to KalshiClient.get_candles (RSA-signed, with
    series→historical fallback) and wraps the raw candles in the training-JSON
    shape so the same feature pipeline applies.'''
    client = KalshiClient()

    if end is None:
        end = int(time.time())
    if start is None:
        start = end - 7 * 24 * 3600

    # Kalshi market tickers are <SERIES>-<EVENT>[-<SUBJECT>]; the live
    # /series/{series}/markets/{ticker}/candlesticks endpoint needs the
    # series. Derive it from the first segment when the caller didn't
    # pass one, so live markets don't fall through to the historical
    # endpoint (which only serves settled markets).
    if series_ticker is None:
        series_ticker = ticker.split("-", 1)[0]

    raw_candles = client.get_candles(
        series_ticker=series_ticker, ticker=ticker,
        start=start, end=end, period=period,
    )

    formatted = []
    for c in raw_candles:
        ts = c.get("end_period_ts", 0)
        ya = c.get("yes_ask") or {}
        yb = c.get("yes_bid") or {}
        pr = c.get("price")   or {}
        formatted.append({
            "ds":               pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "end_period_ts":    ts,
            "close":            _to_float(ya.get("close")),
            "high":             _to_float(ya.get("high")),
            "low":              _to_float(ya.get("low")),
            "volume_fp":        str(c.get("volume", 0)),
            "open_interest_fp": str(c.get("open_interest", 0)),
            "yes_ask": {
                "close_dollars": ya.get("close"),
                "high_dollars":  ya.get("high"),
                "low_dollars":   ya.get("low"),
                "open_dollars":  ya.get("open"),
            },
            "yes_bid": {
                "close_dollars": yb.get("close"),
                "high_dollars":  yb.get("high"),
                "low_dollars":   yb.get("low"),
                "open_dollars":  yb.get("open"),
            },
            "price": {"mean_dollars": pr.get("mean")},
            "market_id": ticker,
            "label":     None,
        })

    return {ticker: formatted}


def predict_live(ticker_or_raw,
                 lgbm_model=None, xgb_model=None,
                 scaler=None, feature_cols=None):
    """
    INFERENCE — main inference function.

    Accepts a ticker string (calls Kalshi API) or pre-fetched candle dict.
    Runs the identical feature pipeline as training.
    Returns current YES probability from the most recent candle.

    Args:
        ticker_or_raw  : ticker string OR pre-fetched {ticker: [candles]} dict
        lgbm_model     : trained LightGBM Booster (loaded from disk if None)
        xgb_model      : trained XGBoost Booster  (loaded from disk if None)
        scaler         : fitted StandardScaler     (loaded from disk if None)
        feature_cols   : list of feature names     (loaded from disk if None)

    Returns dict:
        ticker          — market identifier
        current_prob    — current YES probability (ensemble, latest candle)
        hours_to_expiry — hours remaining
        candles_seen    — number of candles used
        all_probs       — probability at each candle (for charting)
        signal          — "YES", "NO", "UNCERTAIN", or "TOO EARLY"
    """
    if lgbm_model is None:
        lgbm_model, xgb_model, scaler, feature_cols = load_models()

    if isinstance(ticker_or_raw, str):
        ticker = ticker_or_raw
        print(f"Fetching live candles for {ticker}...")
        raw = get_and_format_candles(ticker)
    else:
        raw    = ticker_or_raw
        ticker = list(raw.keys())[0]

    # Same pipeline as training — no resolution candle drop at inference
    df = flatten(raw)
    df = preprocess(df)
    df = engineer_features(df)

    cols = [c for c in feature_cols if c in df.columns]
    X    = scaler.transform(df[cols].fillna(-999).values)

    probs = ensemble_predict(lgbm_model, xgb_model, X)

    current_prob    = float(probs[-1])
    hours_remaining = (float(df["hours_to_expiry"].iloc[-1])
                       if "hours_to_expiry" in df.columns else None)
    candles_seen    = len(df)

    # Confidence filter — early candles have sparse features
    if candles_seen < 3:
        signal = "TOO EARLY — need more candles"
    elif current_prob > 0.65:
        signal = "YES"
    elif current_prob < 0.35:
        signal = "NO"
    else:
        signal = "UNCERTAIN"

    result = {
        "ticker":          ticker,
        "current_prob":    round(current_prob, 4),
        "hours_to_expiry": round(hours_remaining, 1) if hours_remaining else None,
        "candles_seen":    candles_seen,
        "all_probs":       [round(float(p), 4) for p in probs],
        "signal":          signal,
    }
    _print_inference_result(result)
    return result


def _print_inference_result(r):
    print(f"\n{'═'*50}")
    print(f"  Ticker         : {r['ticker']}")
    print(f"  Candles seen   : {r['candles_seen']}")
    print(f"  Hours to expiry: {r['hours_to_expiry']}")
    print(f"  YES probability: {r['current_prob']:.1%}")
    print(f"  Signal         : {r['signal']}")
    print(f"{'═'*50}")
    print("  Probability over time:")
    for i, p in enumerate(r["all_probs"]):
        bar = "█" * int(p * 20)
        print(f"    Hour {i+1:>2}: {p:.2%}  {bar}")
