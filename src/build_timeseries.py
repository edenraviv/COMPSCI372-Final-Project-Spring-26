import time
import requests
import pandas as pd
from datetime import datetime
from kalshi_client import KalshiClient
from data_ingestion import write_to_file, read_from_json_file

def to_unix(ts_str: str) -> int:
    '''Convert ISO string → Unix timestamp (int seconds).'''
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return int(dt.timestamp())

def candles_to_df(response):
    '''Convert candles response → DataFrame.'''
    if isinstance(response, dict):
        candles = response.get("candlesticks", [])  # was "candles"
    else:
        candles = response

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["ds"] = pd.to_datetime(df["end_period_ts"], unit="s")

    df["close"] = df["yes_ask"].apply(lambda x: float(x.get("close_dollars", 0)))
    df["high"] = df["yes_ask"].apply(lambda x: float(x.get("high_dollars", 0)))
    df["low"] = df["yes_ask"].apply(lambda x: float(x.get("low_dollars", 0)))

    return df.sort_values("ds").reset_index(drop=True)

def build_market_series(client, raw_market, label_map):
    '''Build ONE market time series.'''
    series_ticker = raw_market["event_ticker"]
    ticker = raw_market["ticker"]

    start_str = raw_market.get("open_time")
    end_str = raw_market.get("close_time") or raw_market.get("settlement_ts")

    if not start_str or not end_str:
        print(f"  [{ticker}] SKIP: missing timestamps")
        return None

    start_ts = to_unix(start_str)
    end_ts = to_unix(end_str)

    #print(f"  [{ticker}] start={start_ts} end={end_ts} diff={end_ts - start_ts}s")

    if end_ts <= start_ts:
        print(f"  [{ticker}] SKIP: end <= start")
        return None

    if (end_ts - start_ts) < 3600:
        print(f"  [{ticker}] SKIP: window < 1 hour")
        return None

    try:
        response = client.get_candles(
            series_ticker=series_ticker,
            ticker=ticker,
            start=start_ts,
            end=end_ts,
            period=60,
        )
        #print(f"  [{ticker}] raw response: {str(response)[:200]}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in (404, 400):
            print(f"  [{ticker}] SKIP: {e.response.status_code}")
            return None
        raise

    df = candles_to_df(response)

    if df.empty:
        print(f"  [{ticker}] SKIP: empty df")
        return None

    df["market_id"] = ticker
    df["series_id"] = series_ticker
    df["label"] = label_map.get(ticker)

    return df


def build_all(client, raw_markets, processed_markets):
    '''Build ALL markets.'''
    out = {}

    label_map = {
        m["market_id"]: m["label"]
        for m in processed_markets
    }

    for i, m in enumerate(raw_markets):
        ticker = m.get("ticker", "unknown")
        try:
            df = build_market_series(client, m, label_map)
            if df is not None:
                out[ticker] = df
                print(f"[{i+1}/{len(raw_markets)}] Built {ticker} → {len(df)} rows")
        except Exception as e:
            print(f"Failed {ticker}: {e}")

        time.sleep(0.2)

    return out


def save(out, path="data/market_timeseries.json"):
    '''Save time series to JSON file.'''
    serializable = {
        k: v.assign(ds=v["ds"].astype(str)).to_dict(orient="records")
        for k, v in out.items()
    }
    write_to_file(serializable, path)
    print(f"Saved {len(serializable)} time series to {path}")

def run(raw_path: str = "data/raw_market_data.json",
        processed_path: str = "data/processed_market_data.json",
        out_path: str = "data/market_timeseries.json"):
    '''Build hourly candle time series for every raw market and save to JSON.'''
    client = KalshiClient()

    raw_markets = read_from_json_file(raw_path)
    processed_markets = read_from_json_file(processed_path)

    print(f"Raw markets: {len(raw_markets)}")
    print(f"Processed markets: {len(processed_markets)}")

    # Check a sample label_map entry
    label_map = {m["market_id"]: m["label"] for m in processed_markets}
    print(f"Label map size: {len(label_map)}")

    # Check a sample raw market has the keys we need
    sample = raw_markets[0]
    print(f"Sample raw market keys: {list(sample.keys())}")
    print(f"Sample ticker: {sample.get('ticker')}")
    print(f"Sample open_time: {sample.get('open_time')}")
    print(f"Sample close_time: {sample.get('close_time')}")
    print(f"Sample settlement_ts: {sample.get('settlement_ts')}")
    print(f"Sample event_ticker: {sample.get('event_ticker')}")

    ts = build_all(client, raw_markets, processed_markets)
    print(f"\nBuilt {len(ts)} time series")
    save(ts, out_path)
    return out_path


if __name__ == "__main__":
    run()