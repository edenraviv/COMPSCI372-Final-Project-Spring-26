"""
Kalshi Prediction Market — End-to-End Pipeline Entry Point
==========================================================
Runs the full pipeline in order:
  1. Ensure data/ directory exists
  2. populate_datasets.run()  — fetch raw markets from Kalshi API
  3. build_timeseries.run()   — fetch hourly candles for each market
  4. engine.train_pipeline()  — preprocess, train, evaluate, save models

Run from the project root:
    python src/main.py
"""

from pathlib import Path
import populate_datasets
import build_timeseries
import engine


DATA_DIR        = Path("data")
TIMESERIES_PATH = DATA_DIR / "market_timeseries.json"


def ensure_data_dir():
    '''Create the data/ directory if it does not already exist.'''
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Data directory ready: {DATA_DIR.resolve()}")


def main():
    '''Run the full pipeline end-to-end.'''
    print("═" * 60)
    print("  Kalshi Prediction Market — Full Pipeline")
    print("═" * 60)

    # Step 1 — data directory
    print("\n[1/4] Preparing data directory...")
    ensure_data_dir()

    # Step 2 — fetch raw + processed market data
    print("\n[2/4] Populating datasets (Kalshi API)...")
    populate_datasets.run()

    # Step 3 — build hourly candle time series per market
    print("\n[3/4] Building time series (Kalshi candles API)...")
    build_timeseries.run()

    # Step 4 — preprocess, train, evaluate, save models
    print("\n[4/4] Training models...")
    engine.train_pipeline(str(TIMESERIES_PATH))

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
