"""
Kalshi Prediction Market — CLI Inference
=========================================
Run a YES-probability prediction for a single market ticker using the
saved ensemble (LightGBM + XGBoost). Fetches live candles via the Kalshi
API, applies the training feature pipeline, and prints the result.

Requires the trained artifacts at the repo root:
    kalshi_lgbm.txt, kalshi_xgb.json, kalshi_scaler.pkl
(produced by `python src/main.py` or `python src/engine.py`).

Usage (from repo root):
    python src/predict.py TICKER
    python src/predict.py KXPOLITICSMENTION-26FEB18-NATO
"""

import argparse
import sys
from requests.exceptions import HTTPError
from inference import predict_live


def main():
    parser = argparse.ArgumentParser(
        description="Predict YES probability for a Kalshi market ticker.")
    parser.add_argument(
        "series_ticker",
        help="Full Kalshi market ticker — either KXSERIES-YYMMMDD "
             "(e.g. KXVOTEHUBTRUMPUPDOWN-26APR23) or "
             "KXSERIES-YYMMMDD-SUBJECT (e.g. KXTRUMPMENTION-26FEB19-AFRI). "
             "A series ticker on its own (e.g. KXTRUMPREMOVE) will not work.")
    parser.add_argument(
        "market_ticker",
        help="Full Kalshi market ticker — either KXSERIES-YYMMMDD "
             "(e.g. KXVOTEHUBTRUMPUPDOWN-26APR23) or "
             "KXSERIES-YYMMMDD-SUBJECT (e.g. KXTRUMPMENTION-26FEB19-AFRI). "
             "A series ticker on its own (e.g. KXTRUMPREMOVE) will not work.")
    args = parser.parse_args()

    try:
        predict_live(args.market_ticker, args.series_ticker)
    except FileNotFoundError as e:
        print(f"Error: missing trained model artifact — {e}", file=sys.stderr)
        print("Run `python src/main.py` first to train and save models.",
              file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 404:
            print(f"Error: Kalshi API returned 404 for ticker "
                  f"'{args.market_ticker}'.", file=sys.stderr)
        else:
            print(f"Kalshi API error ({status}): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
